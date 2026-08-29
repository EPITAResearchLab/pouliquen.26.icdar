import glob, numpy as np, torch, tqdm

# subsample (200 per category across all .npy files in the directory).
# The resulting tensor shape is (N_categories × 200, 28, 28).

# 1. download quickdraws .npy following https://github.com/googlecreativelab/quickdraw-dataset#get-the-data
# 2. generate a .pt from a subset

quickdraws = None
# SET THIS: path to the directory containing downloaded quickdraw .npy files
for p in tqdm.tqdm(glob.glob("path/to/quickdraw/*.npy")):
    t = torch.tensor(np.load(p).reshape(-1, 28, 28)[:200].astype(float) / 255.)
    quickdraws = t if quickdraws is None else torch.cat((quickdraws, t))

torch.save(quickdraws, "quickdraw_subsamples_more.pt")
