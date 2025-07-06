cd ../../

CUDA_VISIBLE_DEVICES=$1 python main.py \
  --molecule chignolin \
  --state unfolded \
  --force_field protein.ff14SBonlysc \
  --solvent implicit/gbn2 \
  --temperature 300 \
  --time 1_000_000 \
  --platform OpenCL \
  --precision mixed 
