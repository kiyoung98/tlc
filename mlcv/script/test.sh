cd ../

export TZ=Asia/Seoul
CURRENT_DATE=$(date '+%m%d_%H%M%S')
# CURRENT_DATE=0417_010530

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name deeplda \
    ++model.model.layers=[45,100,100,5] \
    ++model.checkpoint=False \
    ++model.checkpoint_name=LARGE_DIM \
    ++steeredmd.simulation.k=200
sleep 1

for k in 300 400 500 600 700;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name deeplda \
        ++model.model.layers=[45,100,100,5] \
        ++model.checkpoint=True \
        ++model.checkpoint_name=LARGE_DIM \
        ++steeredmd.simulation.k=$k
    sleep 1
done