#!/bin/bash

molecule="aldp"
method=$1
ckpt=$2
max_seed=$3
ns=10
step=10000000
# step=50000000

export TZ=Asia/Seoul
datetime=$4
echo $datetime

# for (( seed=0; seed<=max_seed; seed++ )); do
#     base_dir="./simulations/${molecule}/${method}/${ns}/log/${datetime}/${seed}"
    
#     python3 FES_from_State.py \
#         --state ${base_dir}/STATE \
#         --outfile ${base_dir}/fes/fes.dat \
#         --temp 300 \
#         --all_stored
# done

# wait
# echo "All post-processing done!"


python3 main.py \
    --seed $max_seed \
    --plumed ./simulations/${molecule}/${method}/plumed.dat \
    --date ${datetime} \
    --method ${method} \
    --step ${step}

echo "Plots done!"