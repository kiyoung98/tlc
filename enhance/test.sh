#!/bin/bash

molecule="aldp"
method=${1:-DeepLDA}
ckpt=${2:-DeepLDA-10}
ns=10
seed=0

export TZ=Asia/Seoul
datetime=debug
echo $datetime

sigma=(0.05)

for idx in "${!sigma[@]}"; do
    sigma=${sigma[$idx]}
    base_dir="./simulations/${molecule}/${method}/${ns}/log/${datetime}_${sigma}/${seed}"
    mkdir -p "${base_dir}/fes"

    cp "./simulations/${molecule}/${method}/plumed.dat" "${base_dir}/plumed.dat"

    if [[ ! "$method" =~ ^(ref|phi|unbiased)$ ]]; then
        # cp "./simulations/${molecule}/${method}/${ns}/${ckpt}.pt" "${base_dir}/${ckpt}.pt"
        sed -i "/deep: PYTORCH_MODEL/,/^\.\.\./ s|FILE=test\.pt|FILE=./simulations/${molecule}/${method}/${ns}/${ckpt}.pt|g" ${base_dir}/plumed.dat
        sed -i "/deep: PYTORCH_MODEL/,/^\.\.\./! s|FILE=|FILE=${base_dir}/|g" ${base_dir}/plumed.dat
    fi
    # sed -i "s|FILE=|FILE=${base_dir}/|g" ${base_dir}/plumed.dat
    sed -i "s|SIGMA=.*|SIGMA=${sigma}|g" ${base_dir}/plumed.dat
    cat ${base_dir}/plumed.dat

    # $(( idx + 3 ))
    CUDA_VISIBLE_DEVICES=$idx gmx mdrun \
        -s ./simulations/aldp/data/nvt/0.tpr \
        -deffnm ${base_dir} \
        -plumed ${base_dir}/plumed.dat \
        -nsteps 10000000 \
        -ntomp 1 \
        -nb gpu \
        -bonded gpu &
    
    # -nsteps 10000000 \
    
    sleep 1
done

wait
echo "All GROMACS Finished!"

for idx in "${!sigma[@]}"; do
    sigma=${sigma[$idx]}
    base_dir="./simulations/${molecule}/${method}/${ns}/log/${datetime}_${sigma}/${seed}"
    # plumed sum_hills \
    #     --bin 100 \
    #     --hills ${base_dir}/HILLS \
    #     --outfile ${base_dir}/fes/ \
    #     --stride 100
    
    python3 FES_from_State.py \
        --state ${base_dir}/STATE \
        --outfile ${base_dir}/fes \
        --temp 300 \
        --all_stored

done

wait
echo "All post-processing done!"


python3 main.py \
    --plumed ./simulations/${molecule}/${method}/plumed.dat \
    --date ${datetime}_${sigma} \
    --method ${method}