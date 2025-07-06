bash experiments.sh TBG 0505_232520-jit 3

# bash test.sh TBG 0502_191856-jit 1


# for seed in 0 1 2
# do
#     base_dir="./simulations/aldp/TBG/10/log/0429_105517_0.025/${seed}"

#     python3 FES_from_State.py \
#         --state ${base_dir}/STATE \
#         --outfile ${base_dir}/fes/fes.dat \
#         --temp 300 \
#         --all_stored
# done

# python3 main.py \
#     --seed 0 \
#     --plumed ./simulations/aldp/TBG/plumed.dat \
#     --date 0429_105517_0.025 \
#     --method TBG