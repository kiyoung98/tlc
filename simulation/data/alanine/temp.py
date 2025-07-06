import mdtraj as md

# # Read the PDB file
# traj = md.load('c5.pdb')

# # Get indices of non-hydrogen atoms 
# heavy_atoms = []
# for atom in traj.topology.atoms:
#     if atom.element.symbol != 'H':
#         heavy_atoms.append(atom.index+1)

# print("Heavy atom indices:", heavy_atoms)

heavy_atom_list = [2, 5, 6, 7, 9, 11, 15, 16, 17, 19]
atom_cnt = len(heavy_atom_list)
cnt = 1
for i in range(atom_cnt):
    for j in range(i + 1, atom_cnt):
        print(f"d{cnt}: DISTANCE ATOMS={heavy_atom_list[i]},{heavy_atom_list[j]}")
        cnt += 1