cd ../../

time=100_000_000

TEMP=( 100 200 300 400 500 600)
for (( i=0; i<${#TEMP[@]}; i++ ));
do
    CUDA_VISIBLE_DEVICES=$(($i + 2)) python main.py \
    --molecule alanine \
    --state c5 \
    --temperature ${TEMP[i]} \
    --force_field amber99 \
    --solvent tip3p \
    --time $time \
    --platform OpenCL \
    --precision mixed &

    sleep 1
    
    CUDA_VISIBLE_DEVICES=$(($i + 2)) python main.py \
    --molecule alanine \
    --state c7ax \
    --temperature ${TEMP[i]} \
    --force_field amber99 \
    --solvent tip3p \
    --time $time \
    --platform OpenCL \
    --precision mixed &

    sleep 1

    CUDA_VISIBLE_DEVICES=$(($i + 2)) python main.py \
    --molecule alanine \
    --state alpha_P \
    --temperature ${TEMP[i]} \
    --force_field amber99 \
    --solvent tip3p \
    --time $time \
    --platform OpenCL \
    --precision mixed &
done