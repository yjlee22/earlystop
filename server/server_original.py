import os
import time
import numpy as np
import copy
import torch

from utils import *
from dataset import Dataset
from torch.utils import data
from tqdm import trange
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score


class Server(object):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        self.args = args
        self.device = device
        self.datasets = datasets
        self.model_func = model_func
        self.server_model = init_model
        self.server_model_params_list = init_par_list
        self.init_server_model_params_list = copy.deepcopy(init_par_list)

        # (Proposed: growth-rate based)
        self.proposed_early_stop_counter = 0
        self.proposed_early_stop_round = None
        self.prev_distance_from_start = 0.0

        # (Validation-based)
        self.min_val_loss = float('inf')
        self.val_loss_early_stop_counter = 0
        self.val_loss_early_stop_round = None

        self.max_val_acc = 0.0
        self.val_acc_early_stop_counter = 0
        self.val_acc_early_stop_round = None

        print("Initialize the Server      --->  {:s}".format(self.args.method))
        print("Initialize the Public Storage:")
        self.clients_params_list = init_par_list.repeat(args.total_client, 1)
        print("   Local Model Param List  --->  {:d} * {:d}".format(
            self.clients_params_list.shape[0], self.clients_params_list.shape[1]))

        self.clients_updated_params_list = torch.zeros(
            (args.total_client, init_par_list.shape[0]))
        print(" Local Updated Param List  --->  {:d} * {:d}".format(
            self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))

        self.test_perf = np.zeros((self.args.comm_rounds, 5))   # loss, acc, prec, rec, f1
        self.growth_rate_record = np.zeros((self.args.comm_rounds, 1))

        self.time = np.zeros((args.comm_rounds))
        self.lr = self.args.local_learning_rate

        self.comm_vecs = {'Params_list': None}
        self.received_vecs = None
        self.Client = None

        loader_kwargs = dict(
            batch_size=256,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4
        )
        
        self.testloader = data.DataLoader(
            Dataset(self.datasets.test_x, self.datasets.test_y,
                    train=False, dataset_name=self.args.dataset, args=self.args),
            **loader_kwargs
        )

        self.valloader = None
        if getattr(self.args, "validation", False):
            self.valloader = data.DataLoader(
                Dataset(self.datasets.val_x, self.datasets.val_y,
                        train=False, dataset_name=self.args.dataset, args=self.args),
                **loader_kwargs
            )

    def _activate_clients_(self, t):
        return np.random.choice(
            range(self.args.total_client),
            max(int(self.args.active_ratio * self.args.total_client), 1),
            replace=False
        )

    def _lr_scheduler_(self):
        self.lr *= self.args.lr_decay

    def _test_(self, t):
        loss, acc, prec, rec, f1 = self._validate_loader_(self.testloader)
        self.test_perf[t] = [loss, acc, prec, rec, f1]
        print(
            "    Test    ----    Loss: {:.4f},   Acc: {:.4f},    P: {:.4f},    R: {:.4f},    F1: {:.4f}"
            .format(loss, acc, prec, rec, f1),
            flush=True
        )

    def _summary_(self):
        # Determine summary folder based on mode
        if self.args.validation:
            mode_folder = "validation"
        else:  # self.args.proposed
            mode_folder = "proposed"

        if not self.args.non_iid:
            summary_root = f'{self.args.out_file}/summary/{mode_folder}/IID'
        else:
            summary_root = f'{self.args.out_file}/summary/{mode_folder}/{self.args.split_rule}_{self.args.split_coef}'

        if not os.path.exists(summary_root):
            os.makedirs(summary_root)

        valid_perf = self.test_perf[:np.count_nonzero(self.test_perf[:, 1]) + 1]
        if len(valid_perf) == 0:
            valid_perf = self.test_perf

        suffix = f"{self.args.model}_{self.args.pretrain}_{self.args.optimizer}_{self.args.local_learning_rate}_{self.args.patience}_{self.args.threshold}"
        summary_file = summary_root + f'/{self.args.method}_{suffix}.txt'

        with open(summary_file, 'w') as f:
            f.write("##=============================================##\n")
            f.write("##          Early Stopping Records             ##\n")
            f.write("##=============================================##\n")

            if self.args.validation:
                if self.val_loss_early_stop_round is not None:
                    val_loss_round_str = str(self.val_loss_early_stop_round)
                    val_loss_acc = self.test_perf[self.val_loss_early_stop_round][1]
                    val_loss_acc_str = f"{val_loss_acc:.2f}%"
                else:
                    val_loss_round_str = "Not Triggered"
                    val_loss_acc_str = "N/A"

                if self.val_acc_early_stop_round is not None:
                    val_acc_round_str = str(self.val_acc_early_stop_round)
                    val_acc_acc = self.test_perf[self.val_acc_early_stop_round][1]
                    val_acc_acc_str = f"{val_acc_acc:.2f}%"
                else:
                    val_acc_round_str = "Not Triggered"
                    val_acc_acc_str = "N/A"

                f.write(f"Validation (Loss) Stop Round: {val_loss_round_str}\n")
                f.write(f"  - Test Accuracy at Stop   : {val_loss_acc_str}\n")
                f.write(f"Validation (Acc) Stop Round : {val_acc_round_str}\n")
                f.write(f"  - Test Accuracy at Stop   : {val_acc_acc_str}\n")

            else:  # proposed
                if self.proposed_early_stop_round is not None:
                    prop_round_str = str(self.proposed_early_stop_round)
                    prop_acc = self.test_perf[self.proposed_early_stop_round][1]
                    prop_acc_str = f"{prop_acc:.2f}%"
                else:
                    prop_round_str = "Not Triggered"
                    prop_acc_str = "N/A"

                f.write(f"Proposed Early Stop Round   : {prop_round_str}\n")
                f.write(f"  - Test Accuracy at Stop   : {prop_acc_str}\n")

            f.write("##=============================================##\n")

        print("##=============================================##")
        if not self.args.fast:
            print("##                  Summary                    ##")
        else:
            print("##          Summary (Fast Mode)                ##")
        print("##=============================================##")

        if self.args.validation:
            print(f"      Validation (Loss) Stop --->   {val_loss_round_str}")
            print(f"      Validation (Acc) Stop  --->   {val_acc_round_str}")
        else:
            print(f"      Proposed E-Stop        --->   {prop_round_str}")

    def _validate_loader_(self, loader):
        """Evaluate current server_model on a pre-built DataLoader."""
        self.server_model.eval()
        loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

        test_loss = 0.0
        precision_metric, recall_metric, f1_metric, acc_metric = None, None, None, None

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).reshape(-1).long()

                logits = self.server_model(inputs)
                test_loss += loss_func(logits, labels).item()
                preds = logits.argmax(dim=1)

                if precision_metric is None:
                    num_classes = logits.shape[1]
                    precision_metric = Precision(
                        task="multiclass", num_classes=num_classes, average="macro").to(self.device)
                    recall_metric = Recall(
                        task="multiclass", num_classes=num_classes, average="macro").to(self.device)
                    f1_metric = F1Score(
                        task="multiclass", num_classes=num_classes, average="macro").to(self.device)
                    acc_metric = Accuracy(
                        task="multiclass", num_classes=num_classes, average="macro").to(self.device)

                precision_metric.update(preds, labels)
                recall_metric.update(preds, labels)
                f1_metric.update(preds, labels)
                acc_metric.update(preds, labels)

        # handle edge case: empty loader (shouldn't happen)
        denom = (i + 1) if 'i' in locals() else 1

        try:
            acc = acc_metric.compute().item() * 100.0
            precision = precision_metric.compute().item()
            recall = recall_metric.compute().item()
            f1 = f1_metric.compute().item()
        except Exception:
            acc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

        return test_loss / denom, acc, precision, recall, f1

    def _save_results_(self):
        if self.args.fast:
            print("    [FAST MODE] Skipping .npy result saving.")
            return

        if self.args.validation:
            print("    [VALIDATION MODE] Skipping .npy result saving.")
            return

        if not self.args.non_iid:
            root = f'{self.args.out_file}/IID'
        else:
            root = f'{self.args.out_file}/{self.args.split_rule}_{self.args.split_coef}'
        if not os.path.exists(root):
            os.makedirs(root)

        suffix = f"{self.args.model}_{self.args.pretrain}_{self.args.optimizer}_{self.args.local_learning_rate}"
        test_file = root + f'/{self.args.method}_{suffix}.npy'

        final_results = np.hstack((self.test_perf, self.growth_rate_record))
        np.save(test_file, final_results)
        print(f"    Results saved to {test_file} with shape {final_results.shape}")

    # Method-specific hooks
    def process_for_communication(self):
        pass

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        pass

    def postprocess(self, client, received_vecs):
        pass

    def compute_proposed_metrics(self, t):
        """Compute growth rate metrics without checking early stopping condition."""
        dist_vector = self.server_model_params_list - self.init_server_model_params_list
        current_distance_from_start = torch.norm(dist_vector).item()

        if t == 0:
            self.prev_distance_from_start = current_distance_from_start
            self.growth_rate_record[t] = 0.0
            return 0.0

        if self.prev_distance_from_start > 1e-6:
            norm_increase = current_distance_from_start - self.prev_distance_from_start
            growth_rate = norm_increase / self.prev_distance_from_start
        else:
            growth_rate = 1.0

        self.growth_rate_record[t] = growth_rate
        self.prev_distance_from_start = current_distance_from_start
        return growth_rate

    def check_proposed_early_stopping(self, t):
        """Compute metrics and check early stopping condition."""
        growth_rate = self.compute_proposed_metrics(t)

        if self.proposed_early_stop_round is not None:
            return

        if t <= 1 or growth_rate == 0.0:
            return

        if growth_rate < self.args.threshold:
            self.proposed_early_stop_counter += 1
        else:
            self.proposed_early_stop_counter = 0

        if self.proposed_early_stop_counter >= self.args.patience:
            self.proposed_early_stop_round = t

        print(
            f"GR={growth_rate:.4f} | "
            f"cnt={self.proposed_early_stop_counter}/{self.args.patience} | "
            f"stop={self.proposed_early_stop_round}",
            flush=True
        )

    def check_validation_early_stopping(self, t, val_loss, val_acc):
        if self.val_loss_early_stop_round is None:
            if val_loss < (self.min_val_loss - self.args.threshold):
                self.min_val_loss = val_loss
                self.val_loss_early_stop_counter = 0
            else:
                self.val_loss_early_stop_counter += 1

            if self.val_loss_early_stop_counter >= self.args.patience:
                self.val_loss_early_stop_round = t

        if self.val_acc_early_stop_round is None:
            if val_acc > (self.max_val_acc + self.args.threshold):
                self.max_val_acc = val_acc
                self.val_acc_early_stop_counter = 0
            else:
                self.val_acc_early_stop_counter += 1

            if self.val_acc_early_stop_counter >= self.args.patience:
                self.val_acc_early_stop_round = t

    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")

        Averaged_update = torch.zeros(self.server_model_params_list.shape)
        progress_bar = trange(self.args.comm_rounds)

        for t in progress_bar:
            start = time.time()
            selected_clients = self._activate_clients_(t)
            print('============= Communication Round', t + 1, '=============', flush=True)
            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))

            # Client updates
            for client in selected_clients:
                dataset = (self.datasets.client_x[client], self.datasets.client_y[client])
                self.process_for_communication(client, Averaged_update)

                _edge_device = self.Client(
                    device=self.device,
                    model_func=self.model_func,
                    received_vecs=self.comm_vecs,
                    dataset=dataset,
                    lr=self.lr,
                    args=self.args
                )
                self.received_vecs = _edge_device.train()
                self.clients_updated_params_list[client] = self.received_vecs['local_update_list']
                self.clients_params_list[client] = self.received_vecs['local_model_param_list']
                self.postprocess(client, self.received_vecs)
                del _edge_device

            # Aggregation
            Averaged_update = torch.mean(self.clients_updated_params_list[selected_clients], dim=0)
            Averaged_model = torch.mean(self.clients_params_list[selected_clients], dim=0)

            self.server_model_params_list = self.global_update(selected_clients, Averaged_update, Averaged_model)
            set_client_from_params(self.device, self.server_model, self.server_model_params_list)

            # Proposed early stopping
            if self.args.proposed:
                self.check_proposed_early_stopping(t)
            elif not self.args.fast:
                self.compute_proposed_metrics(t)

            # Test
            self._test_(t)

            # Validation (only if enabled)
            if self.args.validation:
                # valloader is created once in __init__
                val_loss, val_acc, _, _, _ = self._validate_loader_(self.valloader)
                self.check_validation_early_stopping(t, val_loss, val_acc)

            # Fast mode stopping
            if self.args.fast:
                should_stop = False

                if self.args.validation:
                    if (self.val_loss_early_stop_round is not None) and (self.val_acc_early_stop_round is not None):
                        should_stop = True
                else:
                    if self.proposed_early_stop_round is not None:
                        should_stop = True

                if should_stop:
                    print(f"\n    >>> [FAST MODE] Early Stopping condition met at round {t}.")
                    end = time.time()
                    self.time[t] = end - start
                    break

            self._lr_scheduler_()
            end = time.time()
            self.time[t] = end - start
            print(f"                ----    Time: {self.time[t]:.2f}s")

        self._save_results_()
        self._summary_()
