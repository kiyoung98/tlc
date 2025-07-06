cd ../../


CUDA_VISIBLE_DEVICES=$1 python main.py \
  --molecule alanine \
  --state c5 \
  --force_field amber99 \
  --solvent tip3p \
  --temperature 300 \
  --time 100_000 \
  --platform OpenCL \
  --precision mixed \
  --seed 0 \
  --log_stdout True \
  --freq_stdout 10 \
  --log_dcd True \
  --freq_dcd 10 \
  --log_csv True \
  --freq_csv 10 \
  --log_force False
