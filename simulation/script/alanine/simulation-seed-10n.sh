cd ../../

for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=$(($seed + 3)) python main.py \
    --molecule alanine \
    --state c5 \
    --force_field amber99 \
    --solvent tip3p \
    --temperature 300 \
    --time 10_000_000 \
    --platform OpenCL \
    --precision mixed \
    --seed $seed \
    --log_stdout True \
    --freq_stdout 1000000 \
    --log_dcd True \
    --freq_dcd 100 \
    --log_csv True \
    --freq_csv 100 \
    --log_force False &
    
    sleep 70
done

for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=$(($seed + 3)) python main.py \
    --molecule alanine \
    --state c7ax \
    --force_field amber99 \
    --solvent tip3p \
    --temperature 300 \
    --time 10_000_000 \
    --platform OpenCL \
    --precision mixed \
    --seed $seed \
    --log_stdout True \
    --freq_stdout 10000 \
    --log_dcd True \
    --freq_dcd 100 \
    --log_csv True \
    --freq_csv 100 \
    --log_force False &
    
    sleep 70
done