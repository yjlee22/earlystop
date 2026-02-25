import os
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
import medmnist
from PIL import Image
from medmnist.info import INFO

class DatasetObject:
    def __init__(self, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path='', args=None):
        self.args = args
        self.n_client = n_client
        self.rule = rule
        self.rule_arg = rule_arg
        self.seed = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = f"{args.dataset.lower()}/seed_{self.seed}_{self.rule}_{rule_arg_str}"
        self.data_path = data_path
        self.width = 64
        self.height = 64
        self.medmnist_root = os.path.join('/work', ".medmnist")
        os.makedirs(self.medmnist_root, exist_ok=True)
        self.set_data()
       
    def set_data(self):
        save_path = os.path.join(self.data_path, 'Data', self.name)
        client_x_path = os.path.join(save_path, 'client_x.npy')
        client_y_path = os.path.join(save_path, 'client_y.npy')
        val_x_path = os.path.join(save_path, 'val_x.npy')
        val_y_path = os.path.join(save_path, 'val_y.npy')
        test_x_path = os.path.join(save_path, 'test_x.npy')
        test_y_path = os.path.join(save_path, 'test_y.npy')

        if not (os.path.exists(client_x_path) and os.path.exists(client_y_path) and 
                os.path.exists(test_x_path) and os.path.exists(test_y_path) and
                os.path.exists(val_x_path) and os.path.exists(val_y_path)):
            
            os.makedirs(save_path, exist_ok=True)
            ds_class_name = self.args.dataset
            DataClass = getattr(medmnist, ds_class_name)

            info = INFO[DataClass.flag]

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            train_dataset = DataClass(split='train', transform=transform, download=True, as_rgb=True, size=64, root=self.medmnist_root)
            val_dataset = DataClass(split='val', transform=transform, download=True, as_rgb=True, size=64, root=self.medmnist_root)
            test_dataset = DataClass(split='test', transform=transform, download=True, as_rgb=True, size=64, root=self.medmnist_root)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=len(train_dataset), shuffle=False, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=True)
            
            self.channels = info['n_channels']
            self.n_cls = len(info['label'])

            train_data = next(iter(train_loader))
            val_data = next(iter(val_loader))
            test_data = next(iter(test_loader))
            
            train_x, train_y = train_data[0].numpy(), train_data[1].numpy().reshape(-1, 1)
            val_x, val_y = val_data[0].numpy(), val_data[1].numpy().reshape(-1, 1)
            test_x, test_y = test_data[0].numpy(), test_data[1].numpy().reshape(-1, 1)

            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(train_y))
            train_x = train_x[rand_perm]
            train_y = train_y[rand_perm]

            self.train_x = train_x
            self.train_y = train_y
            self.val_x = val_x
            self.val_y = val_y
            self.test_x = test_x
            self.test_y = test_y

            n_data_per_client = int(len(train_y) / self.n_client)
            client_data_list = np.ones(self.n_client, dtype=int) * n_data_per_client
            diff = np.sum(client_data_list) - len(train_y)
            
            if diff != 0:
                for client_i in range(self.n_client):
                    if client_data_list[client_i] > diff:
                        client_data_list[client_i] -= diff
                        break

            if self.rule in ['Dirichlet', 'Pathological']:
                if self.rule == 'Dirichlet':
                    cls_priors = np.random.dirichlet(alpha=[self.rule_arg] * self.n_cls, size=self.n_client)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                elif self.rule == 'Pathological':
                    c = int(self.rule_arg)
                    a = np.ones([self.n_client, self.n_cls])
                    a[:, c::] = 0
                    [np.random.shuffle(i) for i in a]
                    prior_cumsum = a.copy()
                    for i in range(prior_cumsum.shape[0]):
                        for j in range(prior_cumsum.shape[1]):
                            if prior_cumsum[i, j] != 0:
                                prior_cumsum[i, j] = a[i, 0:j+1].sum() / c * 1.0

                idx_list = [np.where(train_y == i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                true_sample = [0 for i in range(self.n_cls)]
                
                client_x = [np.zeros((client_data_list[client__], self.channels, self.height, self.width), dtype=np.float32)
                            for client__ in range(self.n_client)]
                client_y = [np.zeros((client_data_list[client__], 1), dtype=np.int64)
                            for client__ in range(self.n_client)]

                while np.sum(client_data_list) != 0:
                    curr_client = np.random.randint(self.n_client)
                    if client_data_list[curr_client] <= 0:
                        continue
                    client_data_list[curr_client] -= 1
                    curr_prior = prior_cumsum[curr_client]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        if cls_amount[cls_label] <= 0:
                            cls_amount[cls_label] = len(idx_list[cls_label])
                            continue
                        cls_amount[cls_label] -= 1
                        true_sample[cls_label] += 1
                        client_x[curr_client][client_data_list[curr_client]] = train_x[idx_list[cls_label][cls_amount[cls_label]]]
                        client_y[curr_client][client_data_list[curr_client]] = train_y[idx_list[cls_label][cls_amount[cls_label]]]
                        break

                print("Sample distribution per class:", true_sample)
                self.client_x = client_x
                self.client_y = client_y

            elif self.rule == 'iid':
                client_x = [np.zeros((client_data_list[client__], self.channels, self.height, self.width), dtype=np.float32)
                            for client__ in range(self.n_client)]
                client_y = [np.zeros((client_data_list[client__], 1), dtype=np.int64)
                            for client__ in range(self.n_client)]
                client_data_list_cum_sum = np.concatenate(([0], np.cumsum(client_data_list)))
                for client_idx_ in range(self.n_client):
                    client_x[client_idx_] = train_x[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                    client_y[client_idx_] = train_y[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                self.client_x = client_x
                self.client_y = client_y

            elif self.rule == 'Quantity':
               
                client_props = np.random.dirichlet(alpha=[self.rule_arg] * self.n_client)

                n_total = len(train_y)
                if n_total < self.n_client:

                    raise ValueError(
                        f"Total training samples ({n_total}) < number of clients ({self.n_client}). "
                        "Decrease n_client or use a dataset with more samples."
                    )

                base_allocation = np.ones(self.n_client, dtype=int)
                remaining = n_total - self.n_client
                extra_allocation = np.random.multinomial(remaining, client_props)
                client_data_list = base_allocation + extra_allocation  # shape: (n_client,)

                client_x = [np.zeros((client_data_list[cid], self.channels, self.height, self.width), dtype=np.float32)
                            for cid in range(self.n_client)]
                client_y = [np.zeros((client_data_list[cid], 1), dtype=np.int64)
                            for cid in range(self.n_client)]

                client_data_list_cum_sum = np.concatenate(([0], np.cumsum(client_data_list)))
                for client_idx_ in range(self.n_client):
                    s = client_data_list_cum_sum[client_idx_]
                    e = client_data_list_cum_sum[client_idx_ + 1]
                    client_x[client_idx_] = train_x[s:e]
                    client_y[client_idx_] = train_y[s:e]

                self.client_x = client_x
                self.client_y = client_y
            
            np.save(client_x_path, np.array(self.client_x, dtype=object))
            np.save(client_y_path, np.array(self.client_y, dtype=object))
            np.save(val_x_path, val_x)
            np.save(val_y_path, val_y)
            np.save(test_x_path, test_x)
            np.save(test_y_path, test_y)
        else:
            self.client_x = np.load(client_x_path, allow_pickle=True)
            self.client_y = np.load(client_y_path, allow_pickle=True)
            self.n_client = len(self.client_x)
            self.val_x  = np.load(val_x_path)
            self.val_y  = np.load(val_y_path)
            self.test_x = np.load(test_x_path)
            self.test_y = np.load(test_y_path)
            ds_class_name = self.args.dataset
            DataClass = getattr(medmnist, ds_class_name)
            info = INFO[DataClass.flag]
            self.channels = info['n_channels']
            self.n_cls = len(info['label'])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y=True, train=False, dataset_name='', args=None):
        self.name = dataset_name
        self.args = args
       
        if self.name == self.args.dataset:
            self.train = train
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            self.X_data = data_x
            self.y_data = data_y if isinstance(data_y, bool) else data_y.astype(np.int64)
            DataClass = getattr(medmnist, self.args.dataset)
            info = INFO[DataClass.flag]
            self.channels = info['n_channels']
        else:
            raise NotImplementedError
           
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        img = self.X_data[idx]
        img = np.moveaxis(img, 0, -1)
        img = self.transform(img)
        return img if isinstance(self.y_data, bool) else (img, self.y_data[idx])

class DatasetFromDir(data.Dataset):
    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]
        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list)