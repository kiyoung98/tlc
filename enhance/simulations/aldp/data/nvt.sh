# for i in {0..7}; do
#     gmx grompp -f nvt.mdp -c A.gro -p topol.top -o nvt_600K/$i.tpr
#     sleep 1
# done

# Generate topology file -p aldp.top
# gmx pdb2gmx \
#     -f c5.gro \
#     -p aldp.top \
#     -ff amber99sbildn \
#     -water tip3p

# Generate tpr file -o nvt.tpr
gmx grompp \
    -f sample.mdp \
    -c c5.gro \
    -p aldp.top \
    -o nvt.tpr
