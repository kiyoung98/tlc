FONTSIZE = 28
FONTSIZE_SMALL = 20
COLORMAP="viridis"
HEX_GRIDSIZE = 30

ALANINE_ATOM_NUM = 22
ALANINE_HEAVY_ATOM_NUM = 10
ALANINE_TBGCV_SCALING = 1.4936519791
ALANINE_CV_BOUND = 0.5

ALDP_PHI_ANGLE = [4, 6, 8, 14]
ALDP_PSI_ANGLE = [6, 8, 14, 16]
ALANINE_BACKBONE_ATOM_IDX = [1, 4, 6, 8, 10, 14, 16, 18]
ALANINE_HEAVY_ATOM_IDX = [1, 4, 5, 6, 8, 10, 14, 15, 16, 18]
ALANINE_HEAVY_ATOM_DESCRIPTOR = [
    "CH3[1]",
    "C[4]",
    "O[5]",
    "N[6]",
    "CA[8]",
    "CB[10]",
    "C[14]",
    "O[15]",
    "N[16]",
    "C[18]",
]

MLCOLVAR_METHODS = [
  "lda",
  "deeplda",
  "deeptda",
  "deeptica",
  "gnncv",
  "autoencoder",
  "vde",
  "vae",
  "tae",
  "tae-xyz",
  "tbgcv",
  "tbgcv-xyz",
  "tbgcv-nolag",
  "tbgcv-xyzhad",
]

MOLECULE = [
  "alanine",
  "chignolin",
]

COLORS = [
    "#A10035",
    "#3FA796",
    "#FEC220",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#D3D3D3", # light gray
]