cd ../

# For jit file
# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name tbgcv-xyz \
#     ++model.model.encoder_layers=[30,100,100,1] \
#     ++model.checkpoint=True \
#     ++model.checkpoint_name=0515_012422 \
#     ++steeredmd.repeat=0 \

# SMD benchmarking
for k in 200 300 400 500;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name tbgcv-xyz \
        ++model.model.encoder_layers=[30,100,100,1] \
        ++model.checkpoint=True \
        ++model.checkpoint_name=0517_234227 \
        ++steeredmd.sample_num=64 \
        ++steeredmd.simulation.k=$k
    sleep 1
done