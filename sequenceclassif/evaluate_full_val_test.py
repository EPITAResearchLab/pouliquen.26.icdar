from glob import glob
from matplotlib import pyplot as plt
import tqdm
from mydatamodules.maeclassif import OVDDetectionModelMAE
from mydatasets.onlinebgsub import *
import random
import json
from os.path import basename
import sys


rois = {
    "ovd": [[[141, 224],[498, 557]]],
    # "ovd1": [[[245, 327], [360, 443]], [[184, 224], [456, 513]], [[141, 224],[498, 557]]],
    "ovd2": [[[900, 145], [1042, 281]]],
    "other": [
        [[0, 0], [786, 236]],
        [[0, 0], [155, 570]],
    ],
}

# for k in range(5):
k = sys.argv[1]
fraud = sys.argv[2] if len(sys.argv) > 2 else "origins"

# SET THIS: absolute path to the rectified dataset root (from Zenodo)
data_path = "data_path/rectified/"
# SET THIS: replace "data_path" with the splits directory root
split_path = f"data_path/data/splits_kfold_s0/k{k}/simple"

runname = "phd_f_final"

j = 15
tmp = glob(f"lightning_logs/version_*/checkpoints/{runname}k{k}-{fraud}-epoch=4-*.ckpt" if fraud != "origins" else f"lightning_logs/version_*/checkpoints/{runname}k{k}-epoch=4-*.ckpt")
while not len(tmp) and j >=0:
    j -= 1
    print(j, f"lightning_logs/version_*/checkpoints/{runname}k{k}-{fraud}-epoch={j}-*.ckpt" if fraud != "origins" else f"lightning_logs/version_*/checkpoints/{runname}k{k}-epoch={j}-*.ckpt")
    tmp = glob(f"lightning_logs/version_*/checkpoints/{runname}k{k}-{fraud}-epoch={j}-*.ckpt" if fraud != "origins" else f"lightning_logs/version_*/checkpoints/{runname}k{k}-epoch={j}-*.ckpt")
model_p = tmp[0]
print("found", model_p)
model = OVDDetectionModelMAE.load_from_checkpoint(model_p)
model.cuda()
model.eval()

# ==========================================================
#                     VAL
# ==========================================================


data_dir = (Path(split_path) / "val.txt").open().read().splitlines()

all_dirs = [
    f"{data_path}/origins"
]

datasets = {}

dir_name = "other_ovd"

data_dirs = [f"{data_path}/origins/{d}/" for d in data_dir]# if d.startswith("ID")]

datasets[dir_name] = OnlineBGSubtractionDataset(
    data_dirs=data_dirs,
    rois=rois,
    category="ovd2",
    sequence_length=5,  # Fixed sequence length
    window_size=10,      # Use sliding window for bg subtraction
    bg_algo="MEDIAN",
    diff_method="classical",
    dataset_labels=0,
    # diff_method="classical",
    # pre_bg_transforms=create_realistic_pre_bg_transforms(),
    post_bg_transforms=create_post_valid_bg_transforms(size=224),
    return_all_sequences=True,
    sequence_stride=5,
    return_original=False, 
)

for dirs in all_dirs:
    dir_name = basename(dirs)

    data_dirs = [f"{dirs}/{d}/" for d in data_dir]# if d.startswith("ID")]

    datasets[dir_name] = OnlineBGSubtractionDataset(
        data_dirs=data_dirs,
        rois=rois,
        category="ovd",
        sequence_length=5,  # Fixed sequence length
        window_size=10,      # Use sliding window for bg subtraction
        bg_algo="MEDIAN",
        diff_method="classical",
        dataset_labels="origins" in dirs,
        # diff_method="classical",
        # pre_bg_transforms=create_realistic_pre_bg_transforms(),
        post_bg_transforms=create_post_valid_bg_transforms(size=224),
        return_all_sequences=True,
        sequence_stride=5,
        return_original=False, 
    )

# ress = batch_inference_multiple_datasets(model, datasets, batch_size=32, num_samples=300)
ress = {}
with torch.no_grad():
    for dataset_name, dataset in tqdm.tqdm(datasets.items(), desc="Processing datasets: "):
        ress[dataset_name] = []
        for i in tqdm.tqdm(range(len(dataset)),
                            desc="random seq: ",
                            leave=False):
            sample = dataset[i]
            res = model(sample["sequence"].unsqueeze(0).cuda())
            # print(f"Valid {res.softmax(-1).cpu()[0]}")
            ress[dataset_name].append([res.softmax(-1).cpu()[0][0].item(), sample["sequence_idx"]])
    
print()
print(f"===== k{k} ==== VAL SET")
print(model_p)
print("Val Mean ± std", {k: f"{np.mean(v)} ± {np.std(v)}" for k, v in ress.items()})
print()
with open(f"jsonres/k{k}-{runname}_{fraud}_full_val.json", "w") as f:
    json.dump(ress, f)


# ================================================
#                    TEST
# ================================================

data_dir = (Path(split_path) / "test.txt").open().read().splitlines()

all_dirs = glob(f"{data_path}/fraud/*") + [f"{data_path}/origins"]

datasets = {}

dir_name = "other_ovd"

data_dirs = [f"{data_path}/origins/{d}/" for d in data_dir]# if d.startswith("ID")]

datasets[dir_name] = OnlineBGSubtractionDataset(
    data_dirs=data_dirs,
    rois=rois,
    category="ovd2",
    sequence_length=5,  # Fixed sequence length
    window_size=10,      # Use sliding window for bg subtraction
    bg_algo="MEDIAN",
    diff_method="classical",
    dataset_labels=0,
    # diff_method="classical",
    # pre_bg_transforms=create_realistic_pre_bg_transforms(),
    post_bg_transforms=create_post_valid_bg_transforms(size=224),
    return_all_sequences=True,
    sequence_stride=5,
    return_original=False,  # Also return original for comparison
)

for dirs in all_dirs:
    dir_name = basename(dirs)

    data_dirs = [f"{dirs}/{d}/" for d in data_dir]# if d.startswith("ID")]

    datasets[dir_name] = OnlineBGSubtractionDataset(
        data_dirs=data_dirs,
        rois=rois,
        category="ovd",
        sequence_length=5,  # Fixed sequence length
        window_size=10,      # Use sliding window for bg subtraction
        bg_algo="MEDIAN",
        diff_method="classical",
        dataset_labels="origins" in dirs,
        # diff_method="classical",
        # pre_bg_transforms=create_realistic_pre_bg_transforms(),
        post_bg_transforms=create_post_valid_bg_transforms(size=224),
        return_all_sequences=True,
        sequence_stride=5,
        return_original=False,  # Also return original for comparison
    )

# ress = batch_inference_multiple_datasets(model, datasets, batch_size=32, num_samples=300)
ress = {}
with torch.no_grad():
    for dataset_name, dataset in tqdm.tqdm(datasets.items(), desc="Processing datasets: "):
        ress[dataset_name] = []
        for i in tqdm.tqdm(range(len(dataset)),
                            desc="random seq: ",
                            leave=False):
            sample = dataset[i]
            res = model(sample["sequence"].unsqueeze(0).cuda())
            # print(f"Valid {res.softmax(-1).cpu()[0]}")
            ress[dataset_name].append([res.softmax(-1).cpu()[0][0].item(), sample["sequence_idx"]])
    
print()
print(f"===== k{k} ==== TEST SET")
print(model_p)
print("Test Mean ± std", {k: f"{np.mean(v)} ± {np.std(v)}" for k, v in ress.items()})
print()
with open(f"jsonres/k{k}-{runname}_{fraud}_full_test.json", "w") as f:
    json.dump(ress, f)


# print("STD", {k: np.std(v) for k, v in ress.items()})
print("finished")
