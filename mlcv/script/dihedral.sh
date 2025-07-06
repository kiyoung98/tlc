cd ../


CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name dihedral \
    ++model.checkpoint_name=debug \
    ++steeredmd.simulation.k=40

for k in 80 120 160 200 240 280 320 360 400;
do
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --config-name dihedral \
        ++model.checkpoint_name=debug \
        ++steeredmd.simulation.k=$k
    sleep 1
done