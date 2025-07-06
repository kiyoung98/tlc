with open('/home/shpark/prj-cmd/simulation/data/chignolin/unfolded.pdb', 'r') as file:
  lines = file.readlines()

heavy_atom_index = []

for i, line in enumerate(lines[:138]):
  if line.strip()[-1] != 'H':
    heavy_atom_index.append(i)
    
print(heavy_atom_index)