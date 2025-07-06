cd ../

for k in 400 500 600 700;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name tbgcv \
        ++model.model.encoder_layers=[45,30,30,1] \
        ++model.checkpoint_name=0502_004128 \
        ++steeredmd.sample_num=64 \
        ++steeredmd.repeat=1 \
        ++steeredmd.simulation.k=$k
    sleep 1
done