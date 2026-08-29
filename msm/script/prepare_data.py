import argparse
import pickle
import torch
from pathlib import Path

from configs.config import Config
from models.models_init import create_model
from training.trainer import Trainer
from mydatasets.utils import seq_mask
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm



transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

def get_num(x) -> int:
    """Extract frame number from filename like 'frame_0001.jpg'."""
    if isinstance(x, Path):
        name = x.stem
    else:
        name = Path(x).stem
    
    # Try to extract number after underscore
    parts = name.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    
    # Fallback: try to extract any number
    import re
    nums = re.findall(r'\d+', name)
    if nums:
        return int(nums[-1])
    
    return 0

class GenericBackbone(nn.Module):
    """Backbone feature extractor (matching your implementation)."""
    
    def __init__(self, backbone_name: str = "mobilevit_xxs", pretrained: bool = True, 
                 model_path: str = "", num_classes: int = -1, dynamic_img_size=False):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        
        self.backbone = (timm.create_model(
            backbone_name,
            pretrained=(pretrained and not len(model_path)),
            num_classes=num_classes,
        ) if not dynamic_img_size else 
        timm.create_model(
            backbone_name,
            pretrained=(pretrained and not len(model_path)),
            num_classes=num_classes,
            dynamic_img_size=dynamic_img_size,
        )
        )
        if model_path and len(model_path) > 0:
            print(f"Loading checkpoint from {model_path}")
            state_dict = torch.load(model_path, weights_only=False)
            self.load_state_dict(state_dict)
        
        # Get output dimension
        self.embed_dim = self._get_embed_dim()
    
    def _get_embed_dim(self) -> int:
        """Infer embedding dimension from model."""
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            return out.shape[-1]
    
    def forward(self, x):
        return self.backbone(x)

# ============================================================
# Feature Extraction Functions (matching your train_mlm scripts)
# ============================================================

def features_video(p_files: Path, model: nn.Module, 
                   seq_len: int,
                   sequences_step: int,
                   device: str = "cuda") -> torch.Tensor:
    """
    Extract windowed sequences from video frames (matching your features_video).
    
    Returns:
        Tensor of shape [num_windows, seq_len, embed_dim] or empty tensor
    """
    # Load frame paths
    if p_files.is_file():
        with open(p_files) as f:
            imgs_p = f.read().splitlines()
    elif p_files.is_dir():
        imgs_p = sorted([f for f in p_files.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    else:
        return torch.tensor([])
    
    if len(imgs_p) < seq_len:
        return torch.tensor([])
    
    # Load and process images
    images = []
    for p in imgs_p:
        if isinstance(p, Path):
            img_path = p
        else:
            img_path = p_files.parent / p if p_files.is_file() else p_files / p
        images.append(Image.open(str(img_path)).convert('RGB'))
    
    # Extract features
    with torch.no_grad():
        imgs_tensor = torch.zeros(len(images), 3, 224, 224)
        for i, img in enumerate(images):
            imgs_tensor[i] = transform(img)[:3]
        del images
        
        features = model(imgs_tensor.to(device)).cpu()
        del imgs_tensor
        torch.cuda.empty_cache()
    
    # Create sliding windows (checking for continuous frame numbers)
    sequences = []
    i = 0
    len_imgs = len(imgs_p)
    
    while i < len_imgs - seq_len:
        num_cur = get_num(imgs_p[i])
        num_end = get_num(imgs_p[i + seq_len - 1])
        
        # Check if frames are continuous
        if (num_end - num_cur + 1) == seq_len:
            sequences.append(features[i:i + seq_len])
            i += sequences_step
        else:
            i += 1
    
    if not sequences:
        return torch.tensor([])
    
    return torch.stack(sequences)


def features_video_full(p_files: Path, model: nn.Module,
                        min_sequence: int,
                        device: str = "cuda") -> tuple[list[torch.Tensor], list]:
    """
    Extract full continuous sequences from video (matching your features_video_full).
    
    Returns:
        Tuple of (list of sequence tensors, list of paths)
    """
    # Load frame paths
    if p_files.is_file():
        with open(p_files) as f:
            imgs_p = f.read().splitlines()
    elif p_files.is_dir():
        imgs_p = sorted([f for f in p_files.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    else:
        raise ValueError(f"Path doesn't exist: {p_files}")
    
    if len(imgs_p) < min_sequence:
        return [], []
    
    # Get frame numbers
    imgs_num = [get_num(im_p) for im_p in imgs_p]
    
    # Find continuous sequences
    sequences_num = []
    i = 0
    for l in range(1, len(imgs_num)):
        if imgs_num[l] != imgs_num[l - 1] + 1:
            if l - i >= min_sequence:
                sequences_num.append((i, l - 1))
            i = l
    if i != l and l - i + 1 >= min_sequence:
        sequences_num.append((i, l))
    
    # Extract features for each continuous segment
    sequences = []
    sequences_paths = []
    
    for beginning, end in sequences_num:
        paths = imgs_p[beginning:end]
        images = []
        for p in paths:
            if isinstance(p, Path):
                img_path = p
            else:
                img_path = p_files.parent / p if p_files.is_file() else p_files / p
            images.append(Image.open(str(img_path)).convert('RGB'))
        
        with torch.no_grad():
            imgs_tensor = torch.zeros(len(images), 3, 224, 224)
            for i, img in enumerate(images):
                imgs_tensor[i] = transform(img)[:3]
            del images
            
            features = model(imgs_tensor.to(device)).cpu()
            del imgs_tensor
            torch.cuda.empty_cache()
            
            sequences.append(features)
            sequences_paths.append(paths)
    
    return sequences, sequences_paths


# ============================================================
# Dataset Creation Functions
# ============================================================

def get_sequences(real_holo: str, fake_holos: list[str], model: nn.Module,
                  split_test: list[str], seq_len: int,
                  sequences_step: int,
                  device: str = "cuda", limit: int|None = None) -> tuple[dict, list[dict]]:
    """
    Get sequences for real and fake holos (matching your get_sequences).
    
    Returns:
        Tuple of (sequences_dict, list_of_fake_sequences_dicts)
    """
    sequences = {}
    print(real_holo)
    videos = sum([list(Path(rh).iterdir()) for rh in real_holo], [])
    # videos = list(Path(real_holo).iterdir())
    if limit:
        videos = videos[:limit]
    
    for v in tqdm(videos, desc="Processing real videos"):
        seq = features_video(v, model, seq_len, sequences_step, device)
        if seq.numel() > 0:
            sequences[v.stem] = seq
    
    sequences_fakeholo = []
    for fake_type in fake_holos:
        seq_fake = {}
        fake_type_p = Path(fake_type)
        print(f"Processing fake: {fake_type_p}")
        
        for v in tqdm(list(fake_type_p.iterdir()), desc=f"Processing {fake_type_p.name}"):
            if v.stem in split_test:
                seq = features_video(v, model, seq_len, sequences_step, device)
                if seq.numel() > 0:
                    seq_fake[v.stem] = seq
        
        sequences_fakeholo.append(seq_fake)
    
    return sequences, sequences_fakeholo


def get_sequences_full(real_holo: str, model: nn.Module,
                       min_sequence: int,
                       device: str = "cuda", 
                       limit: int|None = None) -> dict[str, list[torch.Tensor]]:
    """
    Get full sequences for all videos (matching your sequences_full extraction).
    
    Returns:
        Dict mapping video_stem to list of sequence tensors
    """
    sequences_full = {}
    videos = list(Path(real_holo).iterdir())
    if limit:
        videos = videos[:limit]
    
    for v in tqdm(videos, desc="Extracting full sequences"):
        seqs, _ = features_video_full(v, model, min_sequence, device)
        if seqs:
            sequences_full[v.stem] = seqs
    
    return sequences_full


# ============================================================
# Save/Load Functions (matching your pickle format)
# ============================================================

def save_sequences(sequences: dict, sequences_fakeholo: list[dict], 
                   splits: list[list, list, list], output_path: str):
    """Save windowed sequences in your format."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump([sequences, sequences_fakeholo, splits], f)
    print(f"Saved sequences to {output_path}")


def save_sequences_full(sequences_full: dict, output_path: str):
    """Save full sequences in your format."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(sequences_full, f)
    print(f"Saved full sequences to {output_path}")


def load_sequences(input_path: str) -> tuple[dict, list[dict], tuple]:
    """Load windowed sequences."""
    with open(input_path, "rb") as f:
        sequences, sequences_fakeholo, splits = pickle.load(f)
    return sequences, sequences_fakeholo, splits


def load_sequences_full(input_path: str) -> dict:
    """Load full sequences."""
    with open(input_path, "rb") as f:
        sequences_full = pickle.load(f)
    if isinstance(sequences_full, list):
        sequences_full = sequences_full[0]
    return sequences_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    print(args)
    
    # Load config
    config = Config.from_yaml(args.config)
    print(config.run_name, config.device)
    print(Path(config.paths.data_dir) / config.run_name / "full_data.pt")
    config.device = args.device

    
    # Set seed
    torch.manual_seed(config.training.seed)

    model = GenericBackbone(backbone_name=config.projector.backbone_name,
                            pretrained=config.projector.pretrained,
                            model_path=config.projector.model_path,
                            num_classes=config.projector.num_classes,
                            dynamic_img_size=config.projector.dynamic_img_size)
    model.eval()
    model.to(args.device)
        
    split_dir = Path(config.paths.splits_dir)
    split_train = [l.split("/")[1] for l in (split_dir / "train.txt").read_text().split()]
    split_val = [l.split("/")[1] for l in (split_dir / "val.txt").read_text().split()]
    split_test = [l.split("/")[1] for l in (split_dir / "test.txt").read_text().split()]

    
    splits = (split_train, split_val, split_test)
    
    real_holo = [f"{config.paths.predata_dir}/origins/ID/", f"{config.paths.predata_dir}/origins/passport/"]

    fake_holo_test = [
        f"{config.paths.predata_dir}/fraud/photo_holo_copy/ID/",
        f"{config.paths.predata_dir}/fraud/photo_holo_copy/passport/",
        f"{config.paths.predata_dir}/fraud/copy_without_holo/ID/",
        f"{config.paths.predata_dir}/fraud/copy_without_holo/passport/",
        f"{config.paths.predata_dir}/fraud/pseudo_holo_copy/ID/",
        f"{config.paths.predata_dir}/fraud/pseudo_holo_copy/passport/",
        f"{config.paths.predata_dir}/fraud/photo_replacement/ID/",
        f"{config.paths.predata_dir}/fraud/photo_replacement/passport/",
        f"{config.paths.predata_dir}/fraud/plastified_lowreflect/ID/",
        f"{config.paths.predata_dir}/fraud/plastified_noholo/ID/",
        f"{config.paths.predata_dir}/fraud/no_holo/ID/",
        f"{config.paths.predata_dir}/fraud/no_holo/passport/",
        f"{config.paths.predata_dir}/fraud/swap/ID/",
        f"{config.paths.predata_dir}/fraud/swap/passport/",
        f"{config.paths.predata_dir}/fraud/swap_three/ID/",
        f"{config.paths.predata_dir}/fraud/plain_holo/ID/",
        f"{config.paths.predata_dir}/fraud/plain_holo/passport/",
        f"{config.paths.predata_dir}/fraud/leaf_holo/ID/",
        f"{config.paths.predata_dir}/fraud/leaf_holo/passport/",
        f"{config.paths.predata_dir}/fraud/double_sticker/ID/",
        f"{config.paths.predata_dir}/fraud/holo_completemask/ID/",
        f"{config.paths.predata_dir}/fraud/holo_star_world/ID/",
        f"{config.paths.predata_dir}/fraud/laser/ID/",
        f"{config.paths.predata_dir}/fraud/plastified_led/ID/"
    ]
    sequences, sequences_fakeholo = get_sequences(
        real_holo, fake_holo_test, model, split_test,
        config.data.seq_len, config.data.sequences_step, args.device,
    )
    print(Path(config.paths.data_dir) / config.run_name / "full_data.pt")
    
    save_sequences(sequences, sequences_fakeholo, splits, Path(config.paths.data_dir) / config.run_name / "full_data.pt")


if __name__ == "__main__":
    main()
