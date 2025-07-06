docker run --gpus all \
    -it \
    --name $1 \
    -v /home/shpark/prj-mlcv/lib/enhance:/mnt/enhance \
    gmx-plumed