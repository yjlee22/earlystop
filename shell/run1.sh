methods=("FedAvg" "FedProx" "FedDyn" "SCAFFOLD" "FedSAM" "FedSpeed" "FedSMOO" "FedGamma" "FedLESAM" "FedWMSAM")
seeds=(0 1 2)
coefficients1=(0.01 0.1 1.0)
coefficients2=(1.0 2.0 3.0)
datasets=("BloodMNIST" "DermaMNIST")
declare -A NUM_CLASS=(
  ["BloodMNIST"]=8
  ["DermaMNIST"]=7
)

# Label skew (Dirichlet)
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        num_class="${NUM_CLASS[$dataset]}"
        for method in "${methods[@]}"; do
            for c in "${coefficients1[@]}"; do
                python3 train.py --proposed --fast --non-iid --dataset ${dataset} --num_class ${num_class} --pretrain --method ${method} --seed ${seed} --split-coef ${c} --threshold 0.01
                python3 train.py --validation --fast --non-iid --dataset ${dataset} --num_class ${num_class} --pretrain --method ${method} --seed ${seed} --split-coef ${c} --threshold 0.01
            done
        done
    done
done

# Label skew (Pathological)
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        num_class="${NUM_CLASS[$dataset]}"
        for method in "${methods[@]}"; do
            for c in "${coefficients2[@]}"; do
                python3 train.py --proposed --fast --non-iid --split-rule Pathological --dataset ${dataset} --num_class ${num_class} --pretrain --method ${method} --seed ${seed} --split-coef ${c} --threshold 0.01
                python3 train.py --validation --fast --non-iid --split-rule Pathological --dataset ${dataset} --num_class ${num_class} --pretrain --method ${method} --seed ${seed} --split-coef ${c} --threshold 0.01
            done
        done
    done
done

# Quantity skew
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        num_class="${NUM_CLASS[$dataset]}"
        for method in "${methods[@]}"; do
            for c in "${coefficients1[@]}"; do
                python3 train.py --proposed --fast --non-iid --split-rule Quantity --dataset ${dataset} --num_class ${num_class} --pretrain --method ${method} --seed ${seed} --split-coef ${c} --threshold 0.01
                python3 train.py --validation --fast --non-iid --split-rule Quantity --dataset ${dataset} --num_class ${num_class} --pretrain --method ${method} --seed ${seed} --split-coef ${c} --threshold 0.01
            done
        done
    done
done