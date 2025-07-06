cd ../

export TZ=Asia/Seoul
CURRENT_DATE=$(date '+%m%d_%H%M%S')
# CURRENT_DATE=

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name vae \
    ++model.checkpoint=False \
    ++model.checkpoint_name=$CURRENT_DATE \
    ++steeredmd.simulation.k=200

wait

for k in 300 400 500;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name vae \
        ++model.checkpoint=True \
        ++model.checkpoint_name=$CURRENT_DATE \
        ++steeredmd.simulation.k=$k
    sleep 1
done