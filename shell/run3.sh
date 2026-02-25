methods=("FedAvg" "FedProx" "FedDyn" "SCAFFOLD" "FedSAM" "FedSpeed" "FedSMOO" "FedGamma" "FedLESAM" "FedWMSAM")
seeds=(0 1 2)
thresholds=(0.005 0.05 0.1)
datasets=("BloodMNIST" "DermaMNIST")
declare -A NUM_CLASS=(
  ["BloodMNIST"]=8
  ["DermaMNIST"]=7
)

for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        num_class="${NUM_CLASS[$dataset]}"
        for method in "${methods[@]}"; do
            python3 train.py --local-learning-rate 0.1 --proposed --fast --non-iid --dataset ${dataset} --num_class ${num_class} --pretrain --method ${method} --seed ${seed} --split-coef 0.1 --threshold 0.1
            python3 train.py --local-learning-rate 0.1 --validation --fast --non-iid --dataset ${dataset} --num_class ${num_class} --pretrain --method ${method} --seed ${seed} --split-coef 0.1 --threshold 0.1
        done
    done
done