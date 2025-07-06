import numpy as np
import matplotlib.pyplot as plt

# ─── user parameters ────────────────────────────────
colvar = "../simulations/aldp/DeepLDA/10/log/0430_042954/0/COLVAR"
kB    = 0.0083144621            # kJ/(mol·K)
T     = 300.0                   # K
kT    = kB * T                  # ≈2.494 kJ/mol
n_bins= 100                     # number of φ bins
# ─────────────────────────────────────────────────────

# load data (assumes "#! FIELDS time … phi … bias" header)
data = np.loadtxt(colvar, comments="#")
times = data[:,0]               # simulation time (ps or steps)
phi   = data[:,2]               # φ (rad)
bias  = data[:,-1]              # last column = opes.bias or metad.bias

# compute reweighting factors: w_i = exp(+bias_i/kT)
w = np.exp(bias / kT)

# build a weighted histogram p(phi)
phi_edges = np.linspace(-np.pi, np.pi, n_bins+1)
hist, edges = np.histogram(phi, bins=phi_edges, weights=w, density=True)
centers = 0.5*(edges[:-1] + edges[1:])

# free energy: F = -kT ln p + shift so minimum = 0
F = -kT * hist
F -= F.min()

# plot
plt.figure(figsize=(6,4))
plt.plot(centers, F, "-k", lw=2)
plt.xlabel(r"$\phi$ (rad)")
plt.ylabel(r"$F(\phi)\;/\;\mathrm{kJ\,mol^{-1}}$")
plt.title("Free‐energy surface along $\phi$")
plt.grid(True)
plt.tight_layout()
plt.savefig("test.png", dpi=300, bbox_inches="tight")
plt.show()