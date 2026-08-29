import torch
import numpy as np
import random
import cv2
import glob
import os
from torch.utils.data import Dataset

class SimplifiedOVDSequenceDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        sequence_length=10,
        background_images=None,
        background_images_dir=None,
        num_patterns=5,
        ovd_patterns=5,
        image_size=(224, 224),
        complexity_params=None,
        transform=None,
        return_frames=True,
        seed=42,
        quickdraw_dir=None
    ):
        """Dataset that generates synthetic OVD sequences on-the-fly.

        Args:
            num_samples: Number of synthetic samples to generate
            sequence_length: Length of each sequence
            background_images: List of background image arrays
            background_images_dir: Directory with background images (optional)
            num_patterns: Number of OVD patterns to use
            image_size: Size of images to generate
            complexity_params: Parameters controlling data complexity
            transform: Optional transform to apply to images
            return_frames: If True, return frames as tensor; otherwise, return as numpy arrays
            seed: Random seed for reproducibility
            quickdraw_dir: Directory containing QuickDraw bitmap files (optional)
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.background_images = background_images
        self.background_images_dir = background_images_dir
        self.num_patterns = max(num_patterns, ovd_patterns)
        self.image_size = image_size
        self.complexity_params = complexity_params or {}
        self.transform = transform
        self.return_frames = return_frames
        self.base_seed = seed
        self.quickdraw_dir = quickdraw_dir

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        # Load background images
        self._load_background_images()

        # Generate or load OVD patterns
        self.patterns = self._load_or_generate_patterns()

        # Create sample info (balanced between legit and fake)
        self.sample_info = self._create_sample_info()

    def _load_background_images(self):
        """Load background images from directory if specified"""
        if self.background_images is None and self.background_images_dir:
            self.background_images = []

            # Look for image files in the directory
            extensions = ['.jpg', '.jpeg', '.png']
            img_paths = []
            for extension in extensions:
                img_paths.extend(glob.glob(os.path.join(self.background_images_dir, "*" + extension)))

            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize image to target size
                    img = cv2.resize(img, self.image_size)
                    self.background_images.append(img)
        elif self.background_images is not None and isinstance(self.background_images, list):
            if len(self.background_images) > 0 and isinstance(self.background_images[0], str):
                # List of file paths
                self.background_images = [
                    cv2.resize(cv2.imread(str(img_p)), self.image_size)
                    for img_p in self.background_images if cv2.imread(str(img_p)) is not None
                ]

        # If no background images were loaded, create solid color backgrounds
        if not self.background_images:
            print("No background images provided. Using solid colors.")
            self.background_images = [
                np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) *
                np.array([random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)],
                      dtype=np.uint8)
                for _ in range(10)  # Create 10 different solid colors
            ]

    def _load_or_generate_patterns(self):
        """Load QuickDraw patterns or generate random ones if no QuickDraw directory provided"""
        patterns = []

        if self.quickdraw_dir:  # and os.path.isdir(self.quickdraw_dir):
            for file_path in self.quickdraw_dir:
                try:
                    # Load QuickDraw bitmap data
                    npz_file = np.load(file_path)
                    bitmaps = npz_file['image']

                    # Select a random bitmap from the file
                    bitmap_idx = random.randint(0, len(bitmaps)-1)
                    bitmap = bitmaps[bitmap_idx].reshape(28, 28)

                    # Convert to RGBA with vibrant colors
                    pattern = self._convert_bitmap_to_ovd(bitmap)
                    patterns.append(pattern)
                except Exception as e:
                    print(f"Error loading QuickDraw file {file_path}: {e}")

        # If we couldn't load enough patterns, generate the rest
        while len(patterns) < self.num_patterns:
            patterns.append(self._get_random_bitmap())

        return patterns

    def _convert_bitmap_to_ovd(self, bitmap):
        """Convert a bitmap to a vibrant OVD pattern with transparency"""
        # Apply rotation
        rotation_type = random.choice([
            cv2.ROTATE_180,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        ])
        bitmap = cv2.rotate(bitmap, rotation_type)

        # Apply dilation to make shapes stronger
        kernel_size = random.choice([3, 5])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        bitmap = cv2.dilate(bitmap, kernel, iterations=random.randint(0, 2))

        # Resize to target size
        bitmap = cv2.resize(bitmap, (200, 200))

        # Convert to RGB
        bitmap_rgb = cv2.cvtColor(bitmap, cv2.COLOR_GRAY2RGB)

        # Convert to HSV to apply vibrant colors
        hsv = cv2.cvtColor(bitmap_rgb, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = hsv[..., 1] + 255  # Increase saturation to maximum
        hsv[..., 0] = hsv[..., 0] + random.randint(0, 360)  # Random hue shift
        bitmap_colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply threshold for clean edges
        thresh = random.randint(100, 200)
        bitmap_colored[bitmap < thresh] = 255

        # Create RGBA image with transparency
        bitmap_rgba = cv2.cvtColor(bitmap_colored, cv2.COLOR_BGR2BGRA)
        bitmap_rgba[:, :, 3][bitmap > thresh] = 100  # Semi-transparent OVD
        bitmap_rgba[:, :, 3][bitmap <= thresh] = 0   # Fully transparent background

        return bitmap_rgba

    def _get_random_bitmap(self):
        """Generate a random bitmap to use as an OVD pattern with bright colors"""
        # Create an empty image
        bitmap = np.zeros((28, 28), dtype=np.uint8)

        # Generate random shapes
        num_shapes = random.randint(3, 7)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'ellipse', 'rectangle', 'line'])
            color = random.randint(150, 255)

            if shape_type == 'circle':
                center = (random.randint(5, 23), random.randint(5, 23))
                radius = random.randint(2, 8)
                cv2.circle(bitmap, center, radius, color, -1)
            elif shape_type == 'ellipse':
                center = (random.randint(5, 23), random.randint(5, 23))
                axes = (random.randint(3, 10), random.randint(3, 10))
                angle = random.randint(0, 180)
                cv2.ellipse(bitmap, center, axes, angle, 0, 360, color, -1)
            elif shape_type == 'rectangle':
                pt1 = (random.randint(2, 15), random.randint(2, 15))
                pt2 = (random.randint(16, 26), random.randint(16, 26))
                cv2.rectangle(bitmap, pt1, pt2, color, -1)
            elif shape_type == 'line':
                pt1 = (random.randint(2, 26), random.randint(2, 26))
                pt2 = (random.randint(2, 26), random.randint(2, 26))
                thickness = random.randint(1, 3)
                cv2.line(bitmap, pt1, pt2, color, thickness)

        return self._convert_bitmap_to_ovd(bitmap)

    def _create_sample_info(self):
        """Create sample information for all sequences"""
        # Balance between legit and fake samples
        num_legit = self.num_samples // 2
        num_fake = self.num_samples - num_legit

        # Generate sample info for legitimate samples
        sample_info = [{
            "id": f"legit_{i}",
            "type": "legit",
            "seed": self.base_seed + i,
            "background_idx": random.randint(0, len(self.background_images) - 1)
        } for i in range(num_legit)]

        # Generate sample info for fake samples
        fake_subtypes = [
            "static",          # No change in pattern
            "no_pattern",      # No hologram pattern
            "shuffle",         # Shuffled frame order
            "random"           # Random frames
        ]

        sample_info.extend({
            "id": f"fake_{i}",
            "type": "fake",
            "seed": self.base_seed + num_legit + i,
            "background_idx": random.randint(0, len(self.background_images) - 1),
            "fake_subtype": random.choice(fake_subtypes)
        } for i in range(num_fake))

        # Shuffle samples
        random.shuffle(sample_info)

        return sample_info

    def _add_pattern_to_background(self, background, pattern, position=None, scale=1.0):
        """Add a transparent pattern (OVD) to the background image"""
        # Use random position if none provided
        if position is None:
            bg_h, bg_w = background.shape[:2]
            pattern_h, pattern_w = pattern.shape[:2]

            # Calculate valid position range
            max_x = max(0, bg_w - int(pattern_w * scale))
            max_y = max(0, bg_h - int(pattern_h * scale))

            position = (
                random.randint(0, max(1, max_x)),
                random.randint(0, max(1, max_y))
            )

        # Resize pattern
        pattern = cv2.resize(pattern, None, fx=scale, fy=scale)
        h, w = pattern.shape[:2]
        y, x = position

        result = background.copy()

        # Calculate valid regions
        y1, y2 = max(0, y), min(background.shape[0], y + h)
        x1, x2 = max(0, x), min(background.shape[1], x + w)
        o_y1, o_y2 = max(0, -y), min(h, background.shape[0] - y)
        o_x1, o_x2 = max(0, -x), min(w, background.shape[1] - x)

        # Skip if no valid region
        if y1 >= y2 or x1 >= x2 or o_y1 >= o_y2 or o_x1 >= o_x2:
            return result

        # Apply alpha blending
        alpha_pattern = pattern[o_y1:o_y2, o_x1:o_x2, 3:] / 255.0
        alpha_background = 1.0 - alpha_pattern

        # Apply the pattern with alpha blending
        for c in range(3):
            result[y1:y2, x1:x2, c] = (
                pattern[o_y1:o_y2, o_x1:o_x2, c] * alpha_pattern[:, :, 0] +
                result[y1:y2, x1:x2, c] * alpha_background[:, :, 0]
            ).astype(np.uint8)

        return result

    def _apply_effects(self, frame, complexity=None, glare_position=None, shadow_state=None):
        """Apply visual effects to the frame."""
        if complexity is None:
            complexity = self.complexity_params

        result = frame.copy()

        # Apply jitter if enabled
        jitter_amount = complexity.get("jitter", 0)
        if jitter_amount > 0:
            h, w = result.shape[:2]
            tx = random.randint(-jitter_amount, jitter_amount)
            ty = random.randint(-jitter_amount, jitter_amount)
            matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            result = cv2.warpAffine(result, matrix, (w, h))

        # Apply projection if enabled
        if complexity.get("projection", False) and random.random() < complexity.get("projection_intensity", 1.0):
            h, w = result.shape[:2]
            src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
            max_distortion = min(h, w) * 0.05
            dst_points = src_points + np.random.uniform(-max_distortion, max_distortion,
                                                    src_points.shape).astype(np.float32)
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            result = cv2.warpPerspective(result, matrix, (w, h))

        # Apply glare effect if enabled
        new_glare_position = glare_position
        if complexity.get("glare", False) and random.random() < complexity.get("glare_intensity", 1.0):
            h, w = result.shape[:2]

            # Initialize glare position if not provided
            if glare_position is None:
                glare_position = (random.randint(-100, w+100), random.randint(-100, h+100))

            # Create a glare mask
            mask = np.zeros((h, w), dtype=np.float32)

            # Create a circular glare
            radius = random.randint(h // 6, h // 3)
            center = (int(glare_position[0] % w), int(glare_position[1] % h))
            cv2.circle(mask, center, radius, 1.0, -1)

            # Blur the mask for soft edges
            mask = cv2.GaussianBlur(mask, (25, 25), 0)

            # Apply the glare with a white/yellow tint
            glare_color = np.array([240, 255, 255], dtype=np.uint8)  # Slight yellow tint
            intensity = random.uniform(0.7, 1.0)

            for c in range(3):
                # Screen blend mode for realistic light
                image_norm = result[:,:,c].astype(np.float32) / 255.0
                glare_norm = (mask * glare_color[c] / 255.0) * intensity

                blend = 1.0 - (1.0 - image_norm) * (1.0 - glare_norm)
                result[:,:,c] = np.clip(blend * 255, 0, 255).astype(np.uint8)

            # Update glare position for next frame
            new_glare_position = (
                glare_position[0] + random.randint(-20, 20),
                glare_position[1] + random.randint(-20, 20)
            )

        return result, new_glare_position, shadow_state

    def _generate_legit_sequence(self, sample_info):
        """Generate a legitimate sequence with smooth transitions."""
        random.seed(sample_info["seed"])
        np.random.seed(sample_info["seed"])

        # Get background image
        background_idx = sample_info["background_idx"] % len(self.background_images)
        background = self.background_images[background_idx].copy()

        frames = []
        labels = []

        # Start with a random pattern
        current_pattern_idx = random.randint(0, self.num_patterns - 1)

        # Random initial position for the OVD
        pattern_position = None  # Will be set randomly on first use
        pattern_scale = random.uniform(0.9, 1.1)

        # Initialize effects states
        glare_position = (random.randint(-100, 400), random.randint(-100, 400))
        shadow_state = None

        remaining_frames = self.sequence_length

        while remaining_frames > 0:
            # Select the next pattern (different from current)
            next_pattern_idx = random.choice([i for i in range(self.num_patterns) if i != current_pattern_idx])
            # print(remaining_frames, current_pattern_idx, next_pattern_idx)

            # Number of frames for this transition
            n_transition = random.randint(3, 5)

            # Create smooth transition values
            transition_values = np.linspace(0, 1, n_transition)
            # print(transition_values)
            # Generate each frame in the transition
            for t in transition_values:
                # Blend patterns for smooth transition
                blended_pattern = self.patterns[current_pattern_idx] * (1 - t) + self.patterns[next_pattern_idx] * t

                # Create a new frame from the background
                frame = background.copy()

                # Apply pattern to background
                if pattern_position is None:
                    pattern_position = None  # Will be set randomly
                elif random.random() < 0.1:
                    # 10% chance to change position slightly
                    bg_h, bg_w = frame.shape[:2]
                    max_x = max(0, bg_w - int(blended_pattern.shape[1] * pattern_scale))
                    max_y = max(0, bg_h - int(blended_pattern.shape[0] * pattern_scale))
                    pattern_position = (
                        random.randint(0, max(1, max_x)),
                        random.randint(0, max(1, max_y))
                    )

                frame = self._add_pattern_to_background(
                    frame, blended_pattern.copy(),
                    position=pattern_position, scale=pattern_scale
                )

                # Apply visual effects
                frame, glare_position, shadow_state = self._apply_effects(
                    frame,
                    glare_position=glare_position,
                    shadow_state=shadow_state
                )

                # Add to sequence
                frames.append(frame)
                labels.append({
                    current_pattern_idx: (1 - t),
                    next_pattern_idx: t,
                    "is_legit": 1.0
                })

                remaining_frames -= 1
                if remaining_frames <= 0:
                    break

            # Update current pattern
            current_pattern_idx = next_pattern_idx

        return frames, labels

    def _generate_fake_sequence(self, sample_info):
        """Generate a fake sequence based on the fake subtype"""
        random.seed(sample_info["seed"])
        np.random.seed(sample_info["seed"])

        fake_subtype = sample_info["fake_subtype"]

        # Get background image
        background_idx = sample_info["background_idx"] % len(self.background_images)
        background = self.background_images[background_idx].copy()

        # Initialize effects states
        glare_position = (random.randint(-100, 400), random.randint(-100, 400))
        shadow_state = None

        frames = []
        labels = []

        if fake_subtype == "static":
            # Static sequence: same frame repeated
            pattern_idx = random.randint(0, self.num_patterns - 1)
            static_frame = background.copy()

            # Maybe add pattern (70% chance)
            if random.random() < 0.7:
                static_frame = self._add_pattern_to_background(
                    static_frame, self.patterns[pattern_idx].copy(),
                    scale=random.uniform(0.9, 1.1)
                )

            # Apply effects once
            static_frame, _, _ = self._apply_effects(static_frame, glare_position=glare_position, shadow_state=shadow_state)

            # Repeat for sequence length
            frames = [static_frame.copy() for _ in range(self.sequence_length)]
            labels = [{pattern_idx: 1.0, "is_legit": 0.0} for _ in range(self.sequence_length)]

        elif fake_subtype == "no_pattern":
            # No pattern sequence: just backgrounds with effects
            for _ in range(self.sequence_length):
                frame = background.copy()
                frame, glare_position, shadow_state = self._apply_effects(
                    frame,
                    glare_position=glare_position,
                    shadow_state=shadow_state
                )
                frames.append(frame)
                labels.append({"is_legit": 0.0})

        elif fake_subtype in {"shuffle", "random"}:
            # First generate a legitimate sequence
            legit_frames, _ = self._generate_legit_sequence({
                "seed": sample_info["seed"],
                "background_idx": background_idx
            })

            if fake_subtype == "shuffle":
                # Shuffle the legitimate frames
                shuffled_indices = list(range(len(legit_frames)))
                random.shuffle(shuffled_indices)
                frames = [legit_frames[i] for i in shuffled_indices]
            else:  # random
                # Create completely random frames
                frames = []
                for _ in range(self.sequence_length):
                    # 50% chance to use a random legit frame, 50% chance for new random frame
                    if random.random() < 0.5 and legit_frames:
                        frames.append(random.choice(legit_frames).copy())
                    else:
                        # Get a random background
                        random_bg_idx = random.randint(0, len(self.background_images) - 1)
                        frame = self.background_images[random_bg_idx].copy()

                        # 70% chance to add a random pattern
                        if random.random() < 0.7:
                            random_pattern_idx = random.randint(0, self.num_patterns - 1)
                            frame = self._add_pattern_to_background(
                                frame, self.patterns[random_pattern_idx].copy(),
                                scale=random.uniform(0.8, 1.2)
                            )

                        # Apply effects
                        frame, glare_position, shadow_state = self._apply_effects(
                            frame,
                            glare_position=glare_position,
                            shadow_state=shadow_state
                        )
                        frames.append(frame)

            # Set all labels to fake
            labels = [{"is_legit": 0.0} for _ in range(len(frames))]

            # Ensure we have exactly sequence_length frames
            frames = frames[:self.sequence_length]
            labels = labels[:self.sequence_length]

            # If we don't have enough frames, duplicate the last one
            while len(frames) < self.sequence_length:
                frames.append(frames[-1].copy())
                labels.append(labels[-1].copy())

        return frames, labels

    def _generate_sequence(self, sample_info):
        """Generate a sequence based on sample info"""
        label = int(sample_info["type"] == "legit")
        if sample_info["type"] == "legit":
            frames, labels = self._generate_legit_sequence(sample_info)
        else:
            frames, labels = self._generate_fake_sequence(sample_info)
        # Convert to RGB for output
        # frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        frames = [torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) for frame in frames]

        # Apply transform if specified
        if self.transform:
            frames = self.transform(torch.stack(frames))

        # Return as tensor if requested
        if self.return_frames:
            if self.transform:
                frames = torch.stack(frames)
            else:
                frames = torch.stack([
                    torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
                    for frame in frames
                ])
        # print(labels)

        return frames, label

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.sample_info)

    def __getitem__(self, idx):
        """Get a sample from the dataset by index"""
        sample_info = self.sample_info[idx]

        # Generate the sequence
        frames, labels = self._generate_sequence(sample_info)
        # print(type(frames))

        # Extract is_legit from sample info
        is_legit = 1.0 if sample_info["type"] == "legit" else 0.0

        return frames, is_legit
    
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset




def get_frames(path, splitfile=None, fraud=False, skip_n=1):
    valid_frames = []
    other_frames = []
    video_path = Path(path)
    if splitfile:
        print(f"loading for {video_path.name} from", splitfile, f"skipping {skip_n} so {30/skip_n}fps")
        possible_dirs = [video_path / p for p in Path(splitfile).open().read().splitlines()]
    else:
        possible_dirs = list((video_path / "ID").iterdir()) + (
            list((video_path / "passport").iterdir())
            if (video_path / "passport").exists()
            else [])
    for video_path in possible_dirs:
        image_paths = sorted(video_path.glob("*.jpg"), key=str)
        if (len(image_paths) > 20 and (video_path / "valid_frames.npy").exists()):
            # valid_frames_i = np.load(video_path / "valid_frames.npy")
            video_infos = json.load((video_path / "video_infos.json").open())
            if fraud:
                other_frames += [im_p for i, im_p in enumerate(image_paths) if not i % skip_n]
            else:
                valid_frames += [im_p for i, im_p in enumerate(image_paths)
                                if i in video_infos["valid_frames"] and not i % skip_n]

                other_frames += [im_p for i, im_p in enumerate(image_paths)
                                if i in video_infos["too_bright_frames"] and not i % skip_n]
        else:
            valid_frames += image_paths
    return valid_frames, other_frames

def get_video_frame_groups(path: str, n: int, splitfile: str | None = None) -> dict:
    """Processes each video directory independently and returns groups of n consecutive frames for each video, classifying them as valid or non-valid.

    Arguments:
        path: Path to the main directory containing video subdirectories
        n: Number of consecutive frames in each group
        splitfile: Path to a file containing a list of subdirectories to process

    Returns:
        dict: A dictionary where keys are video directory paths and values are tuples of
              (valid_groups, non_valid_groups) for that video. Each group is a list of n
              consecutive frame paths.
    """
    video_path = Path(path)
    results = {}

    # Determine which directories to process
    if splitfile:
        print(f"loading for {video_path.name} from", splitfile)
        possible_dirs = [video_path / p for p in Path(splitfile).open().read().splitlines()]
    else:
        possible_dirs = (list((video_path / "ID").iterdir())
                         if (video_path / "ID").exists()
                         else [])
        possible_dirs += (list((video_path / "passport").iterdir())
                          if (video_path / "passport").exists()
                          else [])

    # Process each video directory independently
    for video_dir in possible_dirs:
        image_paths = sorted(video_dir.glob("*.jpg"), key=str)

        # Skip directories with too few images or missing valid_frames.npy
        # if not (len(image_paths) > 20 or (video_dir / "valid_frames.npy").exists()):
        #     continue

        # # Get valid and non-valid frames for this video
        # video_infos = json.load((video_dir / "video_infos.json").open())

        # valid_frames_indices = video_infos["valid_frames"]


        # Filter valid and other frames for this video
        # valid_frames = [im_p for i, im_p in enumerate(image_paths)
        #                 if i in video_infos["valid_frames"] or i in video_infos["too_black_frames"]]
        valid_frames = image_paths
        # other_frames = [im_p for i, im_p in enumerate(image_paths)
        #                 if i not in video_infos["valid_frames"] and i not in video_infos["too_black_frames"]]

        # Create frame mapping for this video
        frame_mapping = {}
        for frame_p in valid_frames:  # + other_frames:
            frame_num = int(frame_p.stem.split("_")[1])
            frame_mapping[frame_num] = frame_p

        # Create sets for quick lookup
        valid_frame_nums = {int(frame_p.stem.split("_")[1]) for frame_p in valid_frames}

        # Get all frame numbers in sorted order
        all_frame_nums = sorted(frame_mapping.keys())

        valid_groups = []
        non_valid_groups = []

        # Find groups of consecutive frames
        for i in range(0, len(all_frame_nums) - n + 1, 5):
            start_idx = i
            end_idx = i + n - 1

            # Check if frames form a consecutive sequence
            current_sequence = all_frame_nums[start_idx:end_idx + 1]
            if current_sequence[-1] - current_sequence[0] + 1 == n:  # Ensuring consecutive frames
                # Create group of frame paths
                group = [frame_mapping[num] for num in current_sequence]

                # Check if all frames in the group are valid
                if all(int(frame_p.stem.split("_")[1]) in valid_frame_nums for frame_p in group):
                    valid_groups.append(group)
                else:
                    non_valid_groups.append(group)

        # Store results for this video
        results[video_dir] = (valid_groups, non_valid_groups)

    return results


def get_all_frame_groups(path, n, splitfile=None):
    """A wrapper function that processes all videos independently and then returns combined lists of valid and non-valid groups across all videos.

    Arguments:
        path (str): Path to the main directory containing video subdirectories
        n (int): Number of consecutive frames in each group
        splitfile (str, optional): Path to a file containing a list of subdirectories to process

    Returns:
        tuple: (valid_groups, non_valid_groups) where each element is a list of frame groups
               Each group is a list of n consecutive frame paths
    """
    video_results = get_video_frame_groups(path, n, splitfile)

    # Combine results from all videos
    all_valid_groups = []
    all_non_valid_groups = []

    for valid_groups, non_valid_groups in video_results.values():
        all_valid_groups.extend(valid_groups)
        all_non_valid_groups.extend(non_valid_groups)

    return all_valid_groups, all_non_valid_groups


def get_group_paths(data_path, n, split_path, split_name):
    path = f"{data_path}/origins/"
    split_file = f"{split_path}/{split_name}"
    legit_frames, fake_frames = get_all_frame_groups(path, n, splitfile=split_file)

    path = f"{data_path}/fraud/photo_holo_copy"
    tmp_valid, tmp_other = get_all_frame_groups(path, n, splitfile=split_file)
    fake_frames += tmp_valid
    fake_frames += tmp_other

    path = f"{data_path}/fraud/copy_without_holo"
    tmp_valid, tmp_other = get_all_frame_groups(path, n, splitfile=split_file)
    fake_frames += tmp_valid
    fake_frames += tmp_other

    path = f"{data_path}/fraud/pseudo_holo_copy"
    tmp_valid, tmp_other = get_all_frame_groups(path, n, splitfile=split_file)
    fake_frames += tmp_valid
    fake_frames += tmp_other
    return legit_frames, fake_frames


def get_image_paths(data_path, split_path, split_name, legitonly=False, skip_n=1):
    path = f"{data_path}/origins/"
    split_file = f"{split_path}/{split_name}"
    legit_frames, fake_frames = get_frames(path, splitfile=split_file, skip_n=skip_n)

    if not legitonly:
        path = f"{data_path}/fraud/photo_holo_copy"
        tmp_valid, tmp_other = get_frames(path, splitfile=split_file, skip_n=skip_n)
        fake_frames += tmp_valid
        fake_frames += tmp_other

        path = f"{data_path}/fraud/copy_without_holo"
        tmp_valid, tmp_other = get_frames(path, splitfile=split_file, skip_n=skip_n)
        fake_frames += tmp_valid
        fake_frames += tmp_other

        path = f"{data_path}/fraud/pseudo_holo_copy"
        tmp_valid, tmp_other = get_frames(path, splitfile=split_file, skip_n=skip_n)
        fake_frames += tmp_valid
        fake_frames += tmp_other
    return legit_frames, fake_frames


def get_fake_video_frame_groups(path: str, n: int, splitfile: str | None = None) -> dict:
    video_path = Path(path)
    results = {}

    # Determine which directories to process
    if splitfile:
        print(f"loading for {video_path.name} from", splitfile)
        possible_dirs = [video_path / p for p in Path(splitfile).open().read().splitlines()]
    else:
        possible_dirs = (list((video_path / "ID").iterdir())
                         if (video_path / "ID").exists()
                         else [])
        possible_dirs += (list((video_path / "passport").iterdir())
                          if (video_path / "passport").exists()
                          else [])

    # Process each video directory independently
    for video_dir in possible_dirs:
        image_paths = sorted(video_dir.glob("*.jpg"), key=str)

        # Skip directories with too few images or missing valid_frames.npy
        if not (len(image_paths) > 20 or (video_dir / "valid_frames.npy").exists()):
            continue

        # # Get valid and non-valid frames for this video
        video_infos = json.load((video_dir / "video_infos.json").open())

        valid_frames_indices = video_infos["valid_frames"]


        # Filter valid and other frames for this video
        valid_frames = [im_p for i, im_p in enumerate(image_paths)
                        if i in video_infos["valid_frames"]]# or i in video_infos["too_black_frames"]] # TODO should I consider them or not???
        # valid_frames = image_paths
        other_frames = [im_p for i, im_p in enumerate(image_paths)
                        if i not in video_infos["valid_frames"] and i not in video_infos["too_black_frames"]]

        # Create frame mapping for this video
        frame_mapping = {}
        for frame_p in valid_frames + other_frames:
            frame_num = int(frame_p.stem.split("_")[1])
            frame_mapping[frame_num] = frame_p

        # Create sets for quick lookup
        valid_frame_nums = {int(frame_p.stem.split("_")[1]) for frame_p in valid_frames}

        # Get all frame numbers in sorted order
        all_frame_nums = sorted(frame_mapping.keys())

        valid_groups = []
        non_valid_groups = []

        # Find groups of consecutive frames
        for i in range(0, len(all_frame_nums) - n + 1):
            start_idx = i
            end_idx = i + n - 1

            # Check if frames form a consecutive sequence
            current_sequence = all_frame_nums[start_idx:end_idx+1]
            if current_sequence[-1] - current_sequence[0] + 1 == n:  # Ensuring consecutive frames
                # Create group of frame paths
                group = [frame_mapping[num] for num in current_sequence]

                # Check if all frames in the group are valid
                if all(int(frame_p.stem.split("_")[1]) in valid_frame_nums for frame_p in group):
                    valid_groups.append(group)
                else:
                    non_valid_groups.append(group)
        # Store results for this video
        # results[video_dir] = (valid_groups, non_valid_groups)
        n_fakes = (len(valid_groups) - len(non_valid_groups)) // 3
        # shuffle
        random_seq = [random.sample(image_paths, n) for _ in range(n_fakes)]
        static = [[random.choice(image_paths)] * n for _ in range(n_fakes // 2)]
        static_noholo = []
        if video_infos["too_black_frames"]:
            static_noholo = [[image_paths[random.choice(video_infos["too_black_frames"])]] * n for _ in range(n_fakes // 2)]
        if n_fakes > 0:
            shuffle = [random.sample(seq, len(seq)) for seq in random.sample(valid_groups, n_fakes)]
        else:
            shuffle = []
        results[video_dir] = {
            "valid": valid_groups,
            "too_bright": non_valid_groups,
            "shuffle": shuffle,
            "static": static,
            "static_noholo": static_noholo,
            "random": random_seq,
        }

    return results


def get_from_legits(data_path, n, split_path="train.txt"):
    train_groups = get_fake_video_frame_groups(data_path, n, split_path)  # /origins/ ?
    train_groups_l = list(train_groups.values())
    labels = []
    label_names = []
    whole_frames = []
    for lf in train_groups_l:
        whole_frames.extend(lf["valid"])
        labels.extend([1] * len(lf["valid"]))
        label_names.extend(["valid"] * len(lf["valid"]))
        for k, v in lf.items():
            if k != "valid":
                whole_frames.extend(v)
                labels.extend([0] * len(v))
                label_names.extend([k] * len(v))
    return whole_frames, labels, label_names


def get_from_fakes(data_path, n, split_path="train.txt", additional_fakes=[]):
    labels = []
    label_names = []
    whole_frames = []
    for fraud in ["photo_holo_copy",
                  "copy_without_holo",
                  "pseudo_holo_copy"] + additional_fakes:
        train_groups = get_fake_video_frame_groups(f"{data_path}/{fraud}", n, split_path)  #pjoin
        train_groups_l = list(train_groups.values())
        for lf in train_groups_l:
            for k, v in lf.items():
                if True or not k.startswith("static"):
                    whole_frames.extend(v)
                    labels.extend([0] * len(v))
                    label_names.extend([fraud + k] * len(v))
    return whole_frames, labels, label_names


class VideoDatasetMidvHolo(Dataset):
    """Dataset for MIDV-Holo video sequences with a mix of legitimate and fake samples.

    Args:
        data_path: Path to the data directory
        split_path: Path to the split file
        transform: Transforms to apply to the images
        transform_fake: Optional separate transforms for fake images
        include_real_fakes: Whether to include real-world fake samples
        include_synthetic_fakes: Whether to include synthetic fake samples
        fake_ratio: Ratio of fake samples to legitimate samples
        seq_len: Number of consecutive frames in each sequence
        img_size: Size to resize images to
        split: Dataset split to use (train, val, test)
    """
    def __init__(
        self,
        data_path: str,
        split_path: str,
        transform: Optional[Callable] = None,
        transform_fake: Optional[Callable] = None,
        include_real_fakes: bool = True,
        include_synthetic_fakes: bool = True,
        fake_ratio: float = 0.5,
        seq_len: int = 5,
        img_size: int = 224,
        split: str = "train",
        transfo_individual: float = 0.2
    ):
        self.data_path = data_path
        self.split_path = split_path
        self.transform = transform
        self.transform_fake = transform_fake
        self.seq_len = seq_len
        self.img_size = img_size
        self.transfo_individual = transfo_individual

        # Extract the base split path and add the specific split file
        print(split_path)
        if split_path.endswith(".txt"):
            split_dir = str(Path(split_path).parent)
            split_file = f"{split_dir}/{split}.txt"
        else:
            split_file = f"{split_path}/{split}.txt"

        self.frames = []
        self.labels = []
        self.label_names = []

        # Load legitimate samples and synthetic fakes
        legit_frames, legit_labels, legit_names = get_from_legits(
            f"{data_path}/origins/", seq_len, split_file)
        print(len(legit_frames))

        # Add all valid samples
        valid_indices = [i for i, label in enumerate(legit_labels) if label == 1]
        self.frames.extend([legit_frames[i] for i in valid_indices])
        self.labels.extend([legit_labels[i] for i in valid_indices])
        self.label_names.extend([legit_names[i] for i in valid_indices])

        # Track synthetic fake samples separately
        synthetic_indices = [i for i, label in enumerate(legit_labels) if label == 0]
        synthetic_frames = [legit_frames[i] for i in synthetic_indices]
        synthetic_labels = [legit_labels[i] for i in synthetic_indices]
        synthetic_names = [legit_names[i] for i in synthetic_indices]

        # Load real-world fake samples
        real_fake_frames, real_fake_labels, real_fake_names = [], [], []
        if include_real_fakes:
            fraud_path = f"{data_path}/fraud"
            real_fake_frames, real_fake_labels, real_fake_names = get_from_fakes(
                    fraud_path, seq_len, split_file)

        # Determine balancing if needed
        if fake_ratio > 0:
            n_valid = len(valid_indices)
            target_n_fakes = int(n_valid * fake_ratio / (1 - fake_ratio))

            # Combine fake sources based on inclusion flags
            all_fake_frames = []
            all_fake_labels = []
            all_fake_names = []

            if include_synthetic_fakes:
                all_fake_frames.extend(synthetic_frames)
                all_fake_labels.extend(synthetic_labels)
                all_fake_names.extend(synthetic_names)
                print("holo")

            if include_real_fakes:
                all_fake_frames.extend(real_fake_frames)
                all_fake_labels.extend(real_fake_labels)
                all_fake_names.extend(real_fake_names)
            # print(all_fake_frames)
            # Sample fakes if needed
            if len(all_fake_frames) > target_n_fakes > 0:
                indices = random.sample(range(len(all_fake_frames)), target_n_fakes)
                all_fake_frames = [all_fake_frames[i] for i in indices]
                all_fake_labels = [all_fake_labels[i] for i in indices]
                all_fake_names = [all_fake_names[i] for i in indices]

            # Add sampled fakes to the dataset
            self.frames.extend(all_fake_frames)
            self.labels.extend(all_fake_labels)
            self.label_names.extend(all_fake_names)

        # Shuffle the dataset
        indices = list(range(len(self.frames)))
        random.shuffle(indices)
        self.frames = [self.frames[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.label_names = [self.label_names[i] for i in indices]

        print(f"Dataset created with {len(self.frames)} sequences")
        print(f"  - Valid sequences: {sum(1 for label in self.labels if label == 1)}")
        print(f"  - Fake sequences: {sum(1 for label in self.labels if label == 0)}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_sequence = self.frames[idx]
        label = self.labels[idx]
        label_name = self.label_names[idx]

        # Load all images
        images = [Image.open(img_path).convert("RGB") for img_path in frame_sequence]

        # Apply transforms
        if self.transform:
            # Stack images along a new dimension to create a video tensor
            images = torch.stack([transforms.functional.to_image_tensor(img) for img in images])

            # Apply different transform for specific fake types if provided
            if self.transform_fake and "static" in label_name:
                if random.random() < self.transfo_individual:
                    #print(self.transform_fake(images[0]).shape)
                    tmp = [self.transform_fake(im) for im in images]
                    #print([t.shape for t in tmp])
                    video_tensor = torch.stack(tmp)
                else:
                    video_tensor = self.transform_fake(images)
            else:
                if random.random() < self.transfo_individual:
                    video_tensor = torch.stack([self.transform(im) for im in images])
                else:
                    video_tensor = self.transform(images)
        else:
            # Convert to tensor manually if no transform provided
            video_tensor = torch.stack([transforms.functional.to_image_tensor(img) for img in images])

        # return video_tensor, label
        return {"video_tensor": video_tensor,
                "label": label,
               }

class VideoDatasetMidvHoloAdd(Dataset):
    """Dataset for MIDV-Holo video sequences with a mix of legitimate and fake samples.

    Args:
        data_path: Path to the data directory
        split_path: Path to the split file
        transform: Transforms to apply to the images
        transform_fake: Optional separate transforms for fake images
        include_real_fakes: Whether to include real-world fake samples
        include_synthetic_fakes: Whether to include synthetic fake samples
        fake_ratio: Ratio of fake samples to legitimate samples
        seq_len: Number of consecutive frames in each sequence
        img_size: Size to resize images to
        split: Dataset split to use (train, val, test)
    """
    def __init__(
        self,
        data_path: str,
        split_path: str,
        transform: Optional[Callable] = None,
        transform_fake: Optional[Callable] = None,
        include_real_fakes: bool = True,
        include_synthetic_fakes: bool = True,
        fake_ratio: float = 0.5,
        seq_len: int = 5,
        img_size: int = 224,
        split: str = "train",
        transfo_individual: float = 0.2,
        additional_fake: list[str] = [],
    ):
        self.data_path = data_path
        self.split_path = split_path
        self.transform = transform
        self.transform_fake = transform_fake
        self.seq_len = seq_len
        self.img_size = img_size
        self.transfo_individual = transfo_individual

        # Extract the base split path and add the specific split file
        print(split_path)
        if split_path.endswith(".txt"):
            split_dir = str(Path(split_path).parent)
            split_file = f"{split_dir}/{split}.txt"
        else:
            split_file = f"{split_path}/{split}.txt"

        self.frames = []
        self.labels = []
        self.label_names = []
        self.additional_fake = additional_fake

        # Load legitimate samples and synthetic fakes
        legit_frames, legit_labels, legit_names = get_from_legits(
            f"{data_path}/origins/", seq_len, split_file)
        print(len(legit_frames))

        # Add all valid samples
        valid_indices = [i for i, label in enumerate(legit_labels) if label == 1]
        self.frames.extend([legit_frames[i] for i in valid_indices])
        self.labels.extend([legit_labels[i] for i in valid_indices])
        self.label_names.extend([legit_names[i] for i in valid_indices])

        # Track synthetic fake samples separately
        synthetic_indices = [i for i, label in enumerate(legit_labels) if label == 0]
        synthetic_frames = [legit_frames[i] for i in synthetic_indices]
        synthetic_labels = [legit_labels[i] for i in synthetic_indices]
        synthetic_names = [legit_names[i] for i in synthetic_indices]

        # Load real-world fake samples
        real_fake_frames, real_fake_labels, real_fake_names = [], [], []
        if include_real_fakes:
            fraud_path = f"{data_path}/fraud"
            real_fake_frames, real_fake_labels, real_fake_names = get_from_fakes(
                    fraud_path, seq_len, split_file, additional_fake)

        # Determine balancing if needed
        if fake_ratio > 0:
            n_valid = len(valid_indices)
            target_n_fakes = int(n_valid * fake_ratio / (1 - fake_ratio))

            # Combine fake sources based on inclusion flags
            all_fake_frames = []
            all_fake_labels = []
            all_fake_names = []

            if include_synthetic_fakes:
                all_fake_frames.extend(synthetic_frames)
                all_fake_labels.extend(synthetic_labels)
                all_fake_names.extend(synthetic_names)
                print("holo")

            if include_real_fakes:
                all_fake_frames.extend(real_fake_frames)
                all_fake_labels.extend(real_fake_labels)
                all_fake_names.extend(real_fake_names)
            # print(all_fake_frames)
            # Sample fakes if needed
            if len(all_fake_frames) > target_n_fakes > 0:
                indices = random.sample(range(len(all_fake_frames)), target_n_fakes)
                all_fake_frames = [all_fake_frames[i] for i in indices]
                all_fake_labels = [all_fake_labels[i] for i in indices]
                all_fake_names = [all_fake_names[i] for i in indices]

            # Add sampled fakes to the dataset
            self.frames.extend(all_fake_frames)
            self.labels.extend(all_fake_labels)
            self.label_names.extend(all_fake_names)

        # Shuffle the dataset
        indices = list(range(len(self.frames)))
        random.shuffle(indices)
        self.frames = [self.frames[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.label_names = [self.label_names[i] for i in indices]

        print(f"Dataset created with {len(self.frames)} sequences")
        print(f"  - Valid sequences: {sum(1 for label in self.labels if label == 1)}")
        print(f"  - Fake sequences: {sum(1 for label in self.labels if label == 0)}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_sequence = self.frames[idx]
        label = self.labels[idx]
        label_name = self.label_names[idx]

        # Load all images
        images = [Image.open(img_path).convert("RGB") for img_path in frame_sequence]

        # Apply transforms
        if self.transform:
            # Stack images along a new dimension to create a video tensor
            images = torch.stack([transforms.functional.to_image_tensor(img) for img in images])

            # Apply different transform for specific fake types if provided
            if self.transform_fake and "static" in label_name:
                if random.random() < self.transfo_individual:
                    #print(self.transform_fake(images[0]).shape)
                    tmp = [self.transform_fake(im) for im in images]
                    #print([t.shape for t in tmp])
                    video_tensor = torch.stack(tmp)
                else:
                    video_tensor = self.transform_fake(images)
            else:
                if random.random() < self.transfo_individual:
                    video_tensor = torch.stack([self.transform(im) for im in images])
                else:
                    video_tensor = self.transform(images)
        else:
            # Convert to tensor manually if no transform provided
            video_tensor = torch.stack([transforms.functional.to_image_tensor(img) for img in images])

        # return video_tensor, label
        return {"video_tensor": video_tensor,
                "label": label,
               }


class VideoDataset(Dataset):
    """Dataset for video sequences with provided image paths.

    Args:
        sequences_data: Array with shape [num_sequences, n, image_path]
            - num_sequences: number of video sequences
            - n: number of frames in each sequence
            - image_path: path to each frame image
        labels: Array of labels for each sequence
        transform: Optional transforms to apply to images
        img_size: Size to resize images to (height, width)
    """
    def __init__(
        self,
        sequences_data,  # [num_sequences, n, image_path]
        labels,          # [num_sequences]
        transform=None,
        img_size: Tuple[int, int] = (224, 224),
    ):
        self.sequences_data = sequences_data
        self.labels = labels
        self.transform = transform
        self.img_size = img_size

        # Determine the fixed sequence length from the data
        # Assuming all sequences have the same length
        if len(sequences_data) > 0:
            self.seq_length = len(sequences_data[0])
            #print(f"Using fixed sequence length of {self.seq_length} frames")
        else:
            self.seq_length = 0
            print("Warning: Empty dataset")

        # Create default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.sequences_data)

    def __getitem__(self, idx):
        img_paths = self.sequences_data[idx]  # Get all image paths for this sequence
        label = self.labels[idx]

        # Load all images
        images = [Image.open(img_path).convert("RGB") for img_path in img_paths]
        video_tensor = torch.stack([transforms.functional.to_image_tensor(img) for img in images])
        # Apply transforms to the entire sequence at once using v2 transforms
        if self.transform: 
            video_tensor = self.transform(video_tensor)
        
        # return video_tensor, label
        return {"video_tensor": video_tensor,
                "label": label,
               }
    
class VideoLevelDataset(Dataset):
    def __init__(self, data_path="midvholo",
                split_file="splits_kfold_s0/k0/simple/val.txt",
                transform=None):
        self.transform = transform
        origins = get_video_frame_groups(f"{data_path}/origins", 5, split_file)
        self.sequences = []
        self.labels = []
        self.video_ixdtoname = {}
        self.video_label = []
        idx = 0
        for v, seqs in origins.items():
            seq = seqs[0]  # TODO REMOVE
            self.sequences += seq
            self.video_label += [idx] * len(seq)
            self.video_ixdtoname[idx] = str(v)
            idx += 1
        self.labels = [1] * len(self.sequences)
        fraud_names = {"fraud/photo_holo_copy", "fraud/pseudo_holo_copy", "fraud/copy_without_holo"}
        for fraud_name in fraud_names:
            fraud_vids = get_video_frame_groups(f"{data_path}/{fraud_name}", 5, split_file)
            for v, seqs in fraud_vids.items():
                seq = seqs[0]  # TODO REMOVE
                self.sequences += seq
                self.video_label += [idx] * len(seq)
                self.video_ixdtoname[idx] = str(v)
                idx += 1
        self.labels += [0] * (len(self.sequences) - len(self.labels))
        print(f"{len(self.labels)=} {len(self.video_label)=} {len(self.sequences)=}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        video_id = self.video_label[idx]
        
        images = [Image.open(img_path).convert("RGB") for img_path in seq]

        images = torch.stack([transforms.functional.to_image_tensor(img) for img in images])
        # Apply transforms to the entire sequence at once using v2 transforms
        if self.transform:
            images = self.transform(images)
        
        return {"video_tensor": images,
                "label": label,
                "video_id": video_id,
               }
    
from __future__ import annotations
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor
from kornia.contrib import diamond_square
from kornia.augmentation import random_generator as rg
import kornia

class RandomPlasmaShadowBrightness(IntensityAugmentationBase2D):
    def __init__(
        self,
        roughness: Tuple[float, float] = (0.1, 0.7),
        shade_intensity: Tuple[float, float] = (-1.0, 0.0),
        shade_quantity: Tuple[float, float] = (0.0, 1.0),
        intensity: Tuple[float, float] = (1.0, 4.0),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (roughness, "roughness", None, None),
            (shade_intensity, "shade_intensity", None, None),
            (intensity, "intensity", None, None),
            (shade_quantity, "shade_quantity", None, None),
        )

    def apply_transform(
        self, image: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        B, C, H, W = image.shape
        roughness = params["roughness"].to(image)
        shade_intensity = params["shade_intensity"].to(image).view(-1, 1, 1, 1)
        intensity = params["intensity"].to(image).view(-1, 1, 1, 1)
        shade_quantity = params["shade_quantity"].to(image).view(-1, 1, 1, 1)
        brightness_map = intensity * diamond_square((B, C, H, W), roughness, device=image.device, dtype=image.dtype) - 1
        brightness_map *= shade_intensity
        shade_map = diamond_square((B, 1, H, W), roughness, device=image.device, dtype=image.dtype)

        shade_map = (shade_map < shade_quantity).to(image.dtype) * brightness_map
        return (image + shade_map).clamp_(0, 1)

class RandomPlasmaQuickdrawBrightness(IntensityAugmentationBase2D):
    def __init__(
        self,
        roughness: Tuple[float, float] = (0.1, 0.7),
        shade_intensity: Tuple[float, float] = (-1.0, 0.0),
        intensity: Tuple[float, float] = (1.0, 4.0),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (roughness, "roughness", None, None),
            (shade_intensity, "shade_intensity", None, None),
            (intensity, "intensity", None, None),
        )
        self.quickdraw_masks = torch.load("quickdraw_subsamples.pt")
        self.quickdraw_masks = self.quickdraw_masks.reshape(-1, 28, 28) 

    def apply_transform(
        self, image: Tensor, params: Dict[str, Tensor], flags, transform: Optional[Tensor] = None
    ) -> Tensor:
        B, C, H, W = image.shape
        roughness = params["roughness"].to(image)
        shade_intensity = params["shade_intensity"].to(image).view(-1, 1, 1, 1)
        intensity = params["intensity"].to(image).view(-1, 1, 1, 1)
        brightness_map = intensity * diamond_square((B, C, H, W), roughness, device=image.device, dtype=image.dtype) - 1
        brightness_map *= shade_intensity

        # sample quickdraw
        indices = torch.randint(0, len(self.quickdraw_masks), (B,), device=image.device)
        selected_masks = self.quickdraw_masks[indices].to(image.device)

        # Resize all masks at once instead of one-by-one
        mask_batch = transforms.functional.resize(
            selected_masks,
            size=(H, W),
            antialias=True,
        )
        # Ensure mask has right shape [B, 1, H, W]
        if mask_batch.shape[1] != 1:
            mask_batch = mask_batch.unsqueeze(1)

        # Expand mask to match the channels of the brightness map
        mask_batch = mask_batch.expand(-1, C, -1, -1)

        # Apply the mask to the brightness map
        shade_map = mask_batch * brightness_map

        return (image + shade_map).clamp_(0, 1)
    


class PyTorchVideoToSequenceDataset(Dataset):
    def __init__(
        self,
        video_paths: List[str],
        sequence_length: int = 5,
        bg_algo: str = "MEDIAN",
        diff_method: str = "absdiff",
        window_size: int = 0,
        post_bg_transforms: Optional[transforms.Compose] = None,
        label: int = 1,  # Label for "fake" video data
        clip_duration: float = 1.0,  # Duration in seconds for clips
        pytorchvideo_transforms: Optional = None,  # Custom pytorchvideo transforms
        max_clips_per_video: int = 50,  # Safety limit for very long videos
    ):
        import pytorchvideo
        self.video_paths = video_paths
        self.sequence_length = sequence_length
        self.bg_algo = bg_algo
        self.diff_method = diff_method
        self.window_size = window_size
        self.post_bg_transforms = post_bg_transforms
        self.label = label
        self.clip_duration = clip_duration
        self.max_clips_per_video = max_clips_per_video

        self.pv_transforms = pytorchvideo_transforms

        labeled_video_paths = [(path, {"label": label}) for path in video_paths]
        print(labeled_video_paths)
        self.len = len(labeled_video_paths)
        self.dataset = pytorchvideo.data.LabeledVideoDataset(
            labeled_video_paths,
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", 1),
            transform=self.pv_transforms,
            decode_audio=False,
            # decoder="torchvision",
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        """Get item using PyTorchVideo's clip sampler for proper sampling."""
        item = next(iter(self.dataset))

        seq = item["video"]
        # seq = self.perform_bg_subtraction((np.array(seq.permute(1, 2, 3, 0))*255).astype(np.uint8))
        seq = self.perform_bg_subtraction(np.array(seq.permute(1, 2, 3, 0), dtype=np.uint8))

        seq = self.apply_post_bg_augmentations(seq)

        return {
            'sequence': seq,  # Shape: (sequence_length, C, H, W)
            'sequence_idx': item["video_index"],
            'roi_idx': 0,
            # 'variant_idx': clip_idx,
            'roi_coords': [[0, 0], [224, 224]],
            'category': 'video_fake_pv',
            'sequence_length': 5,
            'is_synthetic': True,
            'synthetic_type': 'pytorchvideo_clip_sampler_bg_sub',
            'label': self.label,
            'video_path': item["video_name"],
            'return_all_sequences_mode': False,
        }

    def perform_bg_subtraction(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Perform background subtraction with windowing support."""
        if not len(frames):
            return []

        processed_frames = []

        # Determine window settings
        if self.window_size <= 0 or self.window_size >= len(frames):
            windows = [(0, len(frames))]
        else:
            windows = []
            for i in range(0, len(frames), self.window_size):
                end_idx = min(i + self.window_size, len(frames))
                windows.append((i, end_idx))

        for start_idx, end_idx in windows:
            window_frames = frames[start_idx:end_idx]

            # Compute background
            if self.bg_algo == "MEAN":
                background_frame = np.mean(window_frames, axis=0).astype(np.uint8)
            elif self.bg_algo == "MEDIAN":
                background_frame = np.median(window_frames, axis=0).astype(np.uint8)
            else:
                raise ValueError("Invalid bg_algo. Use 'MEAN' or 'MEDIAN'.")

            max_sv_diff = 0
            sv_differences = []
            frames_diff = []

            # Process each frame in window
            for frame in window_frames:
                # Apply difference method
                if self.diff_method == "absdiff":
                    frame_diff = cv2.absdiff(frame, background_frame)
                elif self.diff_method == "classical":
                    frame_diff = np.clip(frame.astype(float) - background_frame.astype(float), 0, 255).astype(np.uint8)
                else:
                    raise ValueError("Invalid diff_method. Use 'absdiff' or 'classical'.")

                # Convert to HSV and calculate S*V
                hsv_frame = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2HSV)
                sv_diff = hsv_frame[..., 1].astype(float) * hsv_frame[..., 2].astype(float)
                sv_differences.append(sv_diff)
                frames_diff.append(frame_diff)
                max_sv_diff = max(max_sv_diff, np.max(sv_diff))

            # Normalize and enhance
            for i, sv_diff in enumerate(sv_differences):
                combined_final = frames_diff[i].copy().astype(float)
                if max_sv_diff > 0:
                    for j in range(3):
                        combined_final[..., j] *= sv_diff / max_sv_diff
                processed_frames.append(combined_final)

        return processed_frames

    def apply_post_bg_augmentations(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Apply post-background-subtraction augmentations and return as tensor"""
        if self.post_bg_transforms is None:
            # Convert to tensor without transforms
            tensor_frames = []
            for frame in frames:
                frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
                frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                # tensor_frame = transforms.functional.to_image_tensor(pil_image)
                # tensor_frame = transforms.functional.convert_image_dtype(tensor_frame, torch.float32)
                tensor_frame = transforms.functional.to_image(pil_image)
                tensor_frame = transforms.functional.to_dtype(tensor_frame, torch.float32, scale=True)
                tensor_frames.append(tensor_frame)
            return torch.stack(tensor_frames, dim=0)

        # Apply transforms to batch
        tensor_frames = []
        for frame in frames:
            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            # tensor_frame = transforms.functional.to_image_tensor(pil_image)
            # tensor_frame = transforms.functional.convert_image_dtype(tensor_frame, torch.float32)
            tensor_frame = transforms.functional.to_image(pil_image)
            tensor_frame = transforms.functional.to_dtype(tensor_frame, torch.float32, scale=True)
            tensor_frames.append(tensor_frame)

        batch_tensor = torch.stack(tensor_frames, dim=0)
        transformed_batch = self.post_bg_transforms(batch_tensor)

        return transformed_batch