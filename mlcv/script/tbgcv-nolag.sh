cd ../

for k in 100 200 300 400 500 600;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name tbgcv-nolag \
        ++model.model.encoder_layers=[30,100,100,1] \
        ++model.checkpoint=True \
        ++model.checkpoint_name=0512_013355 \
        ++steeredmd.sample_num=64 \
        ++steeredmd.simulation.k=$k
    sleep 1
done