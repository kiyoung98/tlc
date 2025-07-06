export TZ=Asia/Seoul
cd ../

for k in 200 300 400
do
    echo "Running with k = $k"

    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name tae \
        ++data.version=timelag-10n-v1 \
        ++model.checkpoint=False \
        ++steeredmd.simulation.k=$k &

    sleep 1

    CUDA_VISIBLE_DEVICES=$(($1 + 1)) python main.py \
        --config-name tbgcv \
        ++model.model.encoder_layers=[30,30,30,1] \
        ++model.representation=heavy_atom_coordinate \
        ++model.checkpoint=True \
        ++model.checkpoint_name=0502_191856 \
        ++steeredmd.simulation.k=$k &

    wait
done