cd ../

method_list=(deeplda deeptda deeptica tae vde)
k_list=(600 500 400 400 500 500)
ckpt_name=(0507_203059 0507_203106 0508_113022 0508_113040 0509_011004)

for (( i=0; i<${#method_list[@]}; i++ )); do
    method=${method_list[$i]}
    k=${k_list[$i]}

    echo "Plotting ${method}..."
    echo "k: ${k}"
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name ${method} \
        ++model.checkpoint=True \
        ++model.checkpoint_name=${ckpt_name[$i]} \
        ++steeredmd.simulation.k=${k}
done
