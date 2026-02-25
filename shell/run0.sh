seeds=(0 1 2)
datasets=("BloodMNIST" "DermaMNIST")
declare -A NUM_CLASS=(
  ["BloodMNIST"]=8
  ["DermaMNIST"]=7
)
for dataset in "${datasets[@]}"; do
    num_class="${NUM_CLASS[$dataset]}"
    for seed in "${seeds[@]}"; do
        python3 train.py --proposed --non-iid --dataset ${dataset} --num_class ${num_class} --pretrain --method FedAvg --seed ${seed} --split-coef 0.1
        python3 train.py --validation --non-iid --dataset ${dataset} --num_class ${num_class} --pretrain --method FedAvg --seed ${seed} --split-coef 0.1
    done
done