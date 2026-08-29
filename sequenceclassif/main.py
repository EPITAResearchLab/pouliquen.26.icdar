from mydatamodules.maeclassif import OVDDetectionModelMAE
from mydatasets.onlinebgsub import *
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

# SET THIS: absolute path to the rectified dataset root (from Zenodo)
data_path = "rectified/"

# for i in range(5):
i = sys.argv[1] if len(sys.argv) > 1 else 0
print(f"============= fold k{i} ==============")
# SET THIS: replace "path_to_splits" with the actual splits directory root
split_path = f"path_to_splits/data/splits_kfold_s0/k{i}/simple"
print(split_path)
data_onlyorigins = OnlyOrigins(batch_size=64,
                            data_path=data_path,
                            split_path=split_path,
                            num_workers=12)

model = OVDDetectionModelMAE()

checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
    # monitor='train_loss_epoch',
    monitor='val_loss',
    save_top_k=2,
    mode='min',
    filename=f"phd_f_finalk{i}-" + '{epoch}-{val_loss:.2f}'
)

# Initialize trainer with the callback
# trainer = pl.Trainer(max_epochs=3,
trainer = pl.Trainer(max_epochs=3,
                    callbacks=[checkpoint_callback],
                    accelerator="cuda")  # overfit_batches=2)
trainer.fit(model, datamodule=data_onlyorigins)

print("finished training the main model")
