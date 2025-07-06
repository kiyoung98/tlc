cd ../

export TZ=Asia/Seoul
CURRENT_DATE=$(date '+%m%d_%H%M%S')
# CURRENT_DATE=0516_210032

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name tae \
    ++model.checkpoint=False \
    ++model.checkpoint_name=$CURRENT_DATE \
    ++steeredmd.simulation.k=200

wait

for k in 200 300 400 500 600 700 800 900 1000 1100 1200;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name tae \
        ++model.checkpoint=True \
        ++model.checkpoint_name=$CURRENT_DATE \
        ++steeredmd.simulation.k=$k
    sleep 1
done