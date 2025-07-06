import mdtraj as md

pdb = md.load_pdb("c5.pdb")
pdb.save_gro("c5.gro")