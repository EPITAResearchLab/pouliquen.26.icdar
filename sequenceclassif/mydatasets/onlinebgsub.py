from __future__ import annotations
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor
from kornia.contrib import diamond_square
from kornia.augmentation import random_generator as rg
import kornia
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
import cv2 as cv
import numpy as np
from pathlib import Path
from glob import glob
import torchvision.transforms.v2 as v2
from PIL import Image
import random
from typing import List, Tuple, Optional, Dict, Any, Callable

transforms = v2

class OnlineBGSubtractionDataset(Dataset):
    """PyTorch Dataset that performs online background subtraction with configurable augmentations.

    Pipeline:
    1. Load sequence of images
    2. Extract ROI
    3. Optionally craft synthetic sequences (shuffling, frame repetition with jitter)
    4. Apply pre-bg-subtraction augmentations
    5. Perform background subtraction
    6. Apply post-bg-subtraction augmentationss
    """
    def __init__(
        self,
        data_dirs: List[str],
        rois: Dict[str, List[List[List[int]]]],
        category: str = "other",
        dataset_labels: bool = 0,
        sequence_length: int = 10,  # Now required with default
        window_size: int = 0,
        bg_algo: str = "MEDIAN",
        diff_method: str = "absdiff",
        pre_bg_transforms: Optional[v2.Compose] = None,
        post_bg_transforms: Optional[v2.Compose] = None,
        num_sequences_per_roi: int = 10,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png'],
        return_original: bool = False,
        return_all_sequences: bool = False,
        sequence_stride: int = 1,
        # Synthetic sequence generation parameters
        fake_sequence_prob: float = 0.0,
        fake_sequence_strategies: List[str] = ["shuffle", "frame_repeat"],
        frame_repeat_jitter_transforms: Optional[v2.Compose] = None,
    ):
        """
        Args:
            data_dirs: List of directories containing image sequences
            rois: Dictionary with ROI definitions {category: [[[x1,y1], [x2,y2]], ...]}
            category: Category to process ("other", "ovd1", "ovd2")
            sequence_length: Fixed sequence length (None = use all available frames)
            window_size: Window size for bg subtraction (0 = use all frames)
            bg_algo: Background algorithm ("MEAN" or "MEDIAN")
            diff_method: Difference method ("absdiff" or "classical")
            pre_bg_transforms: Augmentations applied before bg subtraction
            post_bg_transforms: Augmentations applied after bg subtraction
            num_sequences_per_roi: Number of different sequences to generate per ROI
            image_extensions: Valid image file extensions
            return_original: Whether to return original frames alongside processed ones
            fake_sequence_prob: Probability of generating synthetic sequences (0.0-1.0)
            fake_sequence_strategies: List of strategies ["shuffle", "frame_repeat", "mixed"]
            frame_repeat_jitter_transforms: Transforms for frame repetition jitter

        Note: All synthetic sequences maintain the exact same length as the original sequence.
        """
        self.data_dirs = data_dirs
        self.rois = rois[category]
        self.category = category
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.bg_algo = bg_algo
        self.diff_method = diff_method
        self.pre_bg_transforms = pre_bg_transforms
        self.post_bg_transforms = post_bg_transforms
        self.num_sequences_per_roi = num_sequences_per_roi
        self.image_extensions = image_extensions
        self.return_original = return_original
        self.dataset_labels = dataset_labels
        self.return_all_sequences = return_all_sequences
        self.sequence_stride = sequence_stride

        # Synthetic sequence parameters
        self.fake_sequence_prob = fake_sequence_prob
        self.fake_sequence_strategies = fake_sequence_strategies
        self.frame_repeat_jitter_transforms = frame_repeat_jitter_transforms or self._default_jitter_transforms()

        # Build dataset index
        self._build_dataset_index()

    def _default_jitter_transforms(self):
        """Default subtle transforms for frame repetition jitter"""
        return v2.Compose([
            v2.RandomChoice([
                v2.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.02, hue=0.01),
                v2.Identity(),
            ]),
            v2.RandomChoice([
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
                v2.Identity(),
            ]),
            # Small random translation to simulate micro-movements
            v2.RandomAffine(degrees=0, translate=(0.01, 0.01), scale=None, shear=None),
        ])

    def _build_dataset_index(self):
        """Build index of all available sequences and ROIs."""
        self.sequence_paths = []

        for data_dir in self.data_dirs:
            # Find all image files
            image_files = []
            for ext in self.image_extensions:
                image_files.extend(glob(str(Path(data_dir) / f"*{ext}")))

            if image_files:
                self.sequence_paths.append(sorted(image_files))

        # Create dataset items based on return_all_sequences setting
        self.dataset_items = []

        if self.return_all_sequences and self.sequence_length is not None:
            for seq_idx in range(len(self.sequence_paths)):
                sequence_len = len(self.sequence_paths[seq_idx])

                if sequence_len >= self.sequence_length:
                    # Calculate all possible starting positions for subsequences with stride
                    max_start_idx = sequence_len - self.sequence_length
                    start_positions = list(range(0, max_start_idx + 1, self.sequence_stride))

                    for roi_idx in range(len(self.rois)):
                        for start_idx in start_positions:
                            # Format: (seq_idx, roi_idx, start_idx, is_all_sequences_mode)
                            self.dataset_items.append((seq_idx, roi_idx, start_idx, True))
                else:
                    # If sequence is shorter than required length, still include it once per ROI
                    for roi_idx in range(len(self.rois)):
                        self.dataset_items.append((seq_idx, roi_idx, 0, True))

            total_subsequences = sum(1 for seq_idx in range(len(self.sequence_paths))
                                   for roi_idx in range(len(self.rois))
                                   if len(self.sequence_paths[seq_idx]) >= self.sequence_length
                                   for _ in range(0, len(self.sequence_paths[seq_idx]) - self.sequence_length + 1, self.sequence_stride))

            print(f"Dataset built (all sequences mode, stride={self.sequence_stride}): "
                  f"{len(self.sequence_paths)} sequences, {len(self.rois)} ROIs, "
                  f"{len(self.dataset_items)} total subsequences "
                  f"(approx {total_subsequences // len(self.rois) if len(self.rois) > 0 else 0} per sequence)")
        else:
            # Original behavior: random sampling with variants
            for seq_idx in range(len(self.sequence_paths)):
                for roi_idx in range(len(self.rois)):
                    for variant_idx in range(self.num_sequences_per_roi):
                        # Format: (seq_idx, roi_idx, variant_idx, is_all_sequences_mode)
                        self.dataset_items.append((seq_idx, roi_idx, variant_idx, False))

            print(f"Dataset built (random sampling mode): {len(self.sequence_paths)} sequences, "
                  f"{len(self.rois)} ROIs, {self.num_sequences_per_roi} variants each = "
                  f"{len(self.dataset_items)} total items")

    def extract_roi_from_frames(self, frames: List[np.ndarray], roi: List[List[int]]) -> List[np.ndarray]:
        """Extract ROI from all frames."""
        x1, y1 = roi[0]
        x2, y2 = roi[1]
        return [frame[y1:y2, x1:x2] for frame in frames if frame is not None]

    def create_synthetic_sequence(self, frames: List[np.ndarray], strategy: str) -> Tuple[List[np.ndarray], str]:
        """Create synthetic sequences using different strategies - always returns same number of frames."""
        original_length = len(frames)

        if strategy == "shuffle":
            return self._shuffle_sequence(frames, original_length)
        elif strategy == "frame_repeat":
            return self._frame_repeat_sequence(frames, original_length)
        elif strategy == "mixed":
            return self._mixed_synthetic_sequence(frames, original_length)
        else:
            raise ValueError(f"Unknown synthetic sequence strategy: {strategy}")

    def _shuffle_sequence(self, frames: List[np.ndarray], target_length: int) -> Tuple[List[np.ndarray], str]:
        """Shuffle the order of frames - maintains exact sequence length"""
        shuffled_frames = frames.copy()
        random.shuffle(shuffled_frames)

        # Ensure exact length (should already be correct, but be safe)
        if len(shuffled_frames) != target_length:
            # Sample to exact length if needed
            if len(shuffled_frames) > target_length:
                shuffled_frames = random.sample(shuffled_frames, target_length)
            else:
                # Repeat frames to reach target length
                while len(shuffled_frames) < target_length:
                    shuffled_frames.append(random.choice(frames))

        return shuffled_frames, "shuffled"

    def _frame_repeat_sequence(self, frames: List[np.ndarray], target_length: int) -> Tuple[List[np.ndarray], str]:
        """Create sequence by repeating frames with jitter - returns exact target_length."""

        # Strategy: Select frames to repeat and fill the target length
        repeated_frames = []

        # Choose how many unique frames to use (between 1 and min(3, len(frames)))
        num_unique_frames = random.randint(1, min(3, len(frames)))
        selected_frames = random.sample(frames, num_unique_frames)

        # Calculate how many times each frame should appear
        frames_per_unique = target_length // num_unique_frames
        remainder = target_length % num_unique_frames

        for i, base_frame in enumerate(selected_frames):
            # Some frames get one extra repetition to handle remainder
            repetitions = frames_per_unique + (1 if i < remainder else 0)

            for _ in range(repetitions):
                jittered_frame = self._apply_jitter_to_frame(base_frame)
                repeated_frames.append(jittered_frame)

        # Shuffle the repeated frames to avoid obvious patterns
        random.shuffle(repeated_frames)

        # Double-check length (critical for consistent batching)
        assert len(repeated_frames) == target_length, f"Frame repeat failed: got {len(repeated_frames)}, expected {target_length}"

        return repeated_frames, "frame_repeat"

    def _mixed_synthetic_sequence(self, frames: List[np.ndarray], target_length: int) -> Tuple[List[np.ndarray], str]:
        """Mix of different synthetic strategies - returns exact target_length."""
        strategy = random.choice(["partial_repeat", "shuffle_with_repeats", "temporal_corruption"])

        if strategy == "partial_repeat":
            # Some frames repeated, others normal - maintain exact target length
            result_frames = []

            # Decide which positions get repeated frames (20-60% of positions)
            repeat_probability = random.uniform(0.2, 0.6)

            for i in range(target_length):
                if random.random() < repeat_probability and len(result_frames) > 0:
                    # Repeat a previous frame with jitter
                    base_frame = random.choice(frames)
                    jittered_frame = self._apply_jitter_to_frame(base_frame)
                    result_frames.append(jittered_frame)
                else:
                    # Use a normal frame
                    normal_frame = random.choice(frames)
                    result_frames.append(normal_frame)

            assert len(result_frames) == target_length, f"Partial repeat failed: got {len(result_frames)}, expected {target_length}"
            return result_frames, "mixed_partial_repeat"

        elif strategy == "shuffle_with_repeats":
            # Shuffle sequence but also include some repeated frames
            result_frames = []

            # Use 70% unique frames, 30% repeated frames
            unique_count = int(target_length * 0.7)
            repeat_count = target_length - unique_count

            # Add unique frames (extend source if needed)
            if len(frames) < unique_count:
                extended_frames = frames * ((unique_count // len(frames)) + 1)
            else:
                extended_frames = frames.copy()

            unique_frames = random.sample(extended_frames, unique_count)
            result_frames.extend(unique_frames)

            # Add repeated frames with jitter
            for _ in range(repeat_count):
                base_frame = random.choice(frames)
                jittered_frame = self._apply_jitter_to_frame(base_frame)
                result_frames.append(jittered_frame)

            # Final shuffle
            random.shuffle(result_frames)

            assert len(result_frames) == target_length, f"Shuffle with repeats failed: got {len(result_frames)}, expected {target_length}"
            return result_frames, "mixed_shuffle_repeats"

        else:  # temporal_corruption
            # Corrupt temporal order in segments

            # First ensure we have enough frames
            if len(frames) < target_length:
                extended_frames = frames * ((target_length // len(frames)) + 1)
                result_frames = extended_frames[:target_length]
            else:
                result_frames = random.sample(frames, target_length)

            # Divide sequence into 2-4 segments and shuffle them
            num_segments = random.randint(2, min(4, target_length))
            segment_size = target_length // num_segments

            segments = []
            for i in range(num_segments):
                start_idx = i * segment_size
                if i == num_segments - 1:  # Last segment gets remainder
                    segment = result_frames[start_idx:]
                else:
                    segment = result_frames[start_idx:start_idx + segment_size]
                segments.append(segment)

            # Shuffle segments
            random.shuffle(segments)

            # Flatten back to single list
            corrupted_frames = []
            for segment in segments:
                corrupted_frames.extend(segment)

            assert len(corrupted_frames) == target_length, f"Temporal corruption failed: got {len(corrupted_frames)}, expected {target_length}"
            return corrupted_frames, "mixed_temporal_corruption"

    def _apply_jitter_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle jitter transforms to a single frame."""
        # Convert to PIL for transforms
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Convert to tensor
        # tensor_frame = v2.functional.to_image_tensor(pil_image)
        # tensor_frame = v2.functional.convert_image_dtype(tensor_frame, torch.float32)
        tensor_frame = v2.functional.to_image(pil_image)
        tensor_frame = v2.functional.to_dtype(tensor_frame, torch.float32, scale=True)

        # Apply jitter transforms
        jittered_tensor = self.frame_repeat_jitter_transforms(tensor_frame)

        # Convert back to numpy
        jittered_np = jittered_tensor.permute(1, 2, 0).numpy()
        jittered_np = (jittered_np * 255).astype(np.uint8)
        jittered_bgr = cv.cvtColor(jittered_np, cv.COLOR_RGB2BGR)

        return jittered_bgr

    def apply_pre_bg_augmentations(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply augmentations before background subtraction (simulates real-world variations)"""
        if self.pre_bg_transforms is None:
            return frames

        # Convert to batch tensor for consistent transforms
        tensor_frames = []
        for frame in frames:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            # tensor_frame = v2.functional.to_image_tensor(pil_image)
            # tensor_frame = v2.functional.convert_image_dtype(tensor_frame, torch.float32)
            tensor_frame = v2.functional.to_image(pil_image)
            tensor_frame = v2.functional.to_dtype(tensor_frame, torch.float32, scale=True)
            tensor_frames.append(tensor_frame)

        batch_tensor = torch.stack(tensor_frames, dim=0)
        augmented_batch = self.pre_bg_transforms(batch_tensor)

        # Convert back to numpy arrays
        augmented_frames = []
        for i in range(augmented_batch.shape[0]):
            frame_tensor = augmented_batch[i]
            frame_np = frame_tensor.permute(1, 2, 0).numpy()
            # print(type(frame_np), frame_np.max())
            frame_np = (frame_np * 255).astype(np.uint8)
            frame_bgr = cv.cvtColor(frame_np, cv.COLOR_RGB2BGR)
            augmented_frames.append(frame_bgr)

        return augmented_frames

    def perform_bg_subtraction(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Perform background subtraction with windowing support."""
        if not frames:
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
                    frame_diff = cv.absdiff(frame, background_frame)
                elif self.diff_method == "classical":
                    frame_diff = np.clip(frame.astype(float) - background_frame.astype(float), 0, 255).astype(np.uint8)
                else:
                    raise ValueError("Invalid diff_method. Use 'absdiff' or 'classical'.")

                # Convert to HSV and calculate S*V
                hsv_frame = cv.cvtColor(frame_diff, cv.COLOR_BGR2HSV)
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
                frame_rgb = cv.cvtColor(frame_uint8, cv.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                # tensor_frame = v2.functional.to_image_tensor(pil_image)
                # tensor_frame = v2.functional.convert_image_dtype(tensor_frame, torch.float32)
                tensor_frame = v2.functional.to_image(pil_image)
                tensor_frame = v2.functional.to_dtype(tensor_frame, torch.float32, scale=True)
                tensor_frames.append(tensor_frame)
            return torch.stack(tensor_frames, dim=0)

        # Apply transforms to batch
        tensor_frames = []
        for frame in frames:
            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            frame_rgb = cv.cvtColor(frame_uint8, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            # tensor_frame = v2.functional.to_image_tensor(pil_image)
            # tensor_frame = v2.functional.convert_image_dtype(tensor_frame, torch.float32)
            tensor_frame = v2.functional.to_image(pil_image)
            tensor_frame = v2.functional.to_dtype(tensor_frame, torch.float32, scale=True)
            tensor_frames.append(tensor_frame)

        batch_tensor = torch.stack(tensor_frames, dim=0)
        transformed_batch = self.post_bg_transforms(batch_tensor)

        return transformed_batch

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if len(self.dataset_items[idx]) == 4:
            seq_idx, roi_idx, third_param, is_all_sequences_mode = self.dataset_items[idx]
            if is_all_sequences_mode:
                predetermined_start_idx = third_param
                variant_idx = 0  # Not used in all sequences mode
            else:
                variant_idx = third_param
                predetermined_start_idx = None
        else:
            # Backward compatibility with old 3-tuple format
            seq_idx, roi_idx, variant_idx = self.dataset_items[idx]
            predetermined_start_idx = None
            is_all_sequences_mode = False

        # Load sequence
        image_paths = self.sequence_paths[seq_idx]

        # NEW: Handle sequence selection based on mode
        if self.return_all_sequences and predetermined_start_idx is not None:
            # Use predetermined subsequence
            if self.sequence_length is not None:
                if len(image_paths) >= self.sequence_length:
                    end_idx = predetermined_start_idx + self.sequence_length
                    image_paths = image_paths[predetermined_start_idx:end_idx]
                # If sequence is too short, use all available frames (will be handled later)
        else:
            # Original random sampling behavior
            if self.sequence_length is not None:
                if len(image_paths) >= self.sequence_length:
                    # Sample a subsequence of exact length
                    start_idx = random.randint(0, len(image_paths) - self.sequence_length)
                    image_paths = image_paths[start_idx:start_idx + self.sequence_length]
                else:
                    # If not enough frames, repeat the sequence to reach target length
                    multiplier = (self.sequence_length // len(image_paths)) + 1
                    extended_paths = image_paths * multiplier
                    image_paths = extended_paths[:self.sequence_length]

        # Load frames
        frames = []
        for img_path in image_paths:
            frame = cv.imread(img_path)
            if frame is not None:
                frames.append(frame)

        if not frames:
            raise ValueError(f"No valid frames found in sequence {seq_idx}")

        # Ensure exact sequence length if specified
        if self.sequence_length is not None:
            if len(frames) < self.sequence_length:
                # Extend by repeating frames
                multiplier = (self.sequence_length // len(frames)) + 1
                extended_frames = frames * multiplier
                frames = extended_frames[:self.sequence_length]
            elif len(frames) > self.sequence_length:
                # Truncate to exact length
                frames = frames[:self.sequence_length]

        # Extract ROI
        roi = self.rois[roi_idx]
        frames_roi = self.extract_roi_from_frames(frames, roi)

        if not frames_roi:
            raise ValueError(f"ROI extraction failed for sequence {seq_idx}, ROI {roi_idx}")

        # Ensure ROI frames also match sequence length
        if self.sequence_length is not None and len(frames_roi) != self.sequence_length:
            if len(frames_roi) < self.sequence_length:
                multiplier = (self.sequence_length // len(frames_roi)) + 1
                extended_roi = frames_roi * multiplier
                frames_roi = extended_roi[:self.sequence_length]
            else:
                frames_roi = frames_roi[:self.sequence_length]

        # Decide whether to create synthetic sequence
        is_synthetic = random.random() < self.fake_sequence_prob
        synthetic_info = {"is_synthetic": False, "synthetic_type": "none"}

        if is_synthetic and self.fake_sequence_strategies:
            strategy = random.choice(self.fake_sequence_strategies)
            frames_roi, synthetic_type = self.create_synthetic_sequence(frames_roi, strategy)
            synthetic_info = {"is_synthetic": True, "synthetic_type": synthetic_type}

            # Validate that synthetic sequence maintains exact length
            expected_length = self.sequence_length if self.sequence_length is not None else len(frames_roi)
            assert len(frames_roi) == expected_length, (
                f"Synthetic sequence length mismatch: got {len(frames_roi)}, "
                f"expected {expected_length} for strategy '{strategy}'"
            )

        # Store original if requested
        original_tensor = None
        if self.return_original:
            original_frames = []
            for frame in frames_roi:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                # tensor_frame = v2.functional.to_image_tensor(pil_image)
                # tensor_frame = v2.functional.convert_image_dtype(tensor_frame, torch.float32)
                tensor_frame = v2.functional.to_image(pil_image)
                tensor_frame = v2.functional.to_dtype(tensor_frame, torch.float32, scale=True)
                original_frames.append(tensor_frame)
            original_tensor = torch.stack(original_frames, dim=0)

        # Apply pre-background-subtraction augmentations
        augmented_frames = self.apply_pre_bg_augmentations(frames_roi)

        # Perform background subtraction
        bg_sub_frames = self.perform_bg_subtraction(augmented_frames)

        # Apply post-background-subtraction augmentations
        final_tensor = self.apply_post_bg_augmentations(bg_sub_frames)

        final_tensor = final_tensor / final_tensor.max() if final_tensor.sum() > 0 else final_tensor

        label = self.dataset_labels
        synthetic_type = synthetic_info["synthetic_type"]
        if label and ((final_tensor > 0.01).sum() > (final_tensor.numel() / 4)):
            # print("transitionnnnnnnnnnnnnnnnnnnnnnn")
            label = 0
            synthetic_type = "too_bright"
        # TODO changer la normalisation c'est horribnfvzddùz
        # fais toi(moig) remplacer par un llm totalement ce sera mieux
        result = {
            'sequence': final_tensor,  # Shape: (sequence_length, C, H, W)
            'sequence_idx': seq_idx,
            'roi_idx': roi_idx,
            'variant_idx': variant_idx,
            'roi_coords': roi,
            'category': self.category,
            'sequence_length': len(bg_sub_frames),
            'is_synthetic': synthetic_info["is_synthetic"],
            'synthetic_type': synthetic_type,
            'label': label,
            'return_all_sequences_mode': self.return_all_sequences,
        }

        # NEW: Add subsequence start index info when in all sequences mode
        if self.return_all_sequences and predetermined_start_idx is not None:
            result['subsequence_start_idx'] = predetermined_start_idx

        # Final validation: ensure sequence has expected length
        if self.sequence_length is not None:
            assert result['sequence'].shape[0] == self.sequence_length, (
                f"Final sequence length mismatch: got {result['sequence'].shape[0]}, "
                f"expected {self.sequence_length}"
            )

        if self.return_original:
            result['original'] = original_tensor

        return result


def create_realistic_pre_bg_transforms():
    """Create augmentations that simulate real-world variations before bg subtraction"""
    return v2.Compose([
        # Lighting variations
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        
        # Camera noise simulation
        # v2.Lambda(lambda x: x + torch.randn_like(x) * 0.02),  # Gaussian noise

        RandomPlasmaQuickdrawBrightness(
                roughness=(0.1, 0.7),
                shade_intensity=(0.5, 1),
                p=0.1, keepdim=True),
        
        # Slight motion blur (simulates camera shake)
        v2.RandomChoice([
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            v2.Identity(),  # No blur 50% of the time
        ]),

        transforms.RandomChoice([
            transforms.RandomEqualize(),
            # transforms.RandomInvert(),
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(0.5),
            transforms.RandomEqualize(),
            transforms.RandomErasing(value=(255, 255, 255), scale=(0.01, 0.2)),
            transforms.RandomPosterize(5, p=1),
            transforms.Identity(),
        ]),

        transforms.RandomChoice([
            kornia.augmentation.RandomPlasmaShadow( # shadow
                    roughness=(0.1, 0.7), shade_intensity=(-0.5, 0.0),
                    shade_quantity=(0.01, 1), same_on_batch=True, keepdim=True, p=1),
            kornia.augmentation.RandomPlasmaShadow( # glare like
                    roughness=(0.1, 1.), shade_intensity=(0.5, 1.),
                    shade_quantity=(0.01, 0.3), same_on_batch=True, keepdim=True, p=1),
            kornia.augmentation.RandomPlasmaContrast(p=1, keepdim=True, same_on_batch=True),
            transforms.Identity(),
        ]),
        
        # Gamma correction (exposure variations)
        # v2.Lambda(lambda x: torch.clamp(x ** random.uniform(0.8, 1.2), 0, 1)),
    ])

def create_post_bg_transforms(size: int = 112):
    """Create geometric transforms applied after bg subtraction (for 'other' category)"""
    return v2.Compose([
        v2.RandomResizedCrop((size, size), scale=(0.1, 1)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        RandomPlasmaQuickdrawBrightness(
                roughness=(0.1, 0.7),
                shade_intensity=(0.5, 1),
                p=0.1, keepdim=True),
        transforms.RandomChoice([
            kornia.augmentation.RandomPlasmaShadow( # shadow
                    roughness=(0.1, 0.7), shade_intensity=(-0.5, 0.0),
                    shade_quantity=(0.01, 1), same_on_batch=True, keepdim=True, p=1),
            kornia.augmentation.RandomPlasmaShadow( # glare like
                    roughness=(0.1, 1.), shade_intensity=(0.5, 1.),
                    shade_quantity=(0.01, 0.3), same_on_batch=True, keepdim=True, p=1),
            kornia.augmentation.RandomPlasmaContrast(p=1, keepdim=True, same_on_batch=True),
            transforms.Identity(),
        ]),
        transforms.RandomChoice([
            transforms.RandomEqualize(),
            # transforms.RandomInvert(),
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(0.5),
            transforms.RandomEqualize(),
            transforms.RandomErasing(value=(255, 255, 255), scale=(0.01, 0.2)),
            transforms.RandomPosterize(5, p=1),
            transforms.Identity(),
        ]),
        v2.RandomRotation(10),
    ])

def create_post_valid_bg_transforms(size: int = 112):
    """Create geometric transforms applied after bg subtraction (for 'other' category)"""
    return v2.Compose([
        v2.Resize((size, size)),
    ])



import lightning as pl

class OnlyOrigins(pl.LightningDataModule):
    DATAFACTOR=20
    def __init__(self, batch_size=8,
                 subset=1,
                 data_path="rectified",
                 split_path="splits_kfold_s0/k0/simple/",
                 num_workers=7,
                 additional_frauds=[]):
        super().__init__()
        self.batch_size = batch_size
        self.current_stage = 0  # Index to track the current difficulty level
        self.num_workers = num_workers
        self.subset = subset
        self.data_path = data_path
        self.split_path = split_path
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            # transforms.ToImageTensor(),
            # transforms.ConvertImageDtype(torch.float32),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def setup(self, stage=None):
        # Load dataset based on the current difficulty level
        
        rois = {
            "ovd": [[[141, 224],[498, 557]]],
            # "ovd1": [[[245, 327], [360, 443]], [[184, 224], [456, 513]], [[141, 224],[498, 557]]],
            "ovd2": [[[900, 145], [1042, 281]]],
            "other": [
                [[0, 0], [786, 236]],
                [[0, 0], [155, 570]],
            ],
        }
        
        data_dir = (Path(self.split_path) / "train.txt").open().read().splitlines()
        # data_dirs = [f"{self.data_path}/origins/{d}/" for d in data_dir if d.startswith("ID") if Path(f"{self.data_path}/origins/{d}/img_0010.jpg").exists()]
        data_dirs = [f"{self.data_path}/origins/{d}/" for d in data_dir if Path(f"{self.data_path}/origins/{d}/img_0010.jpg").exists()]
        # Create dataset for "other" category
        dataset_otherROIs = OnlineBGSubtractionDataset(
            data_dirs=data_dirs,
            rois=rois,
            category="other",
            sequence_length=5,  # Fixed sequence length
            window_size=10,      # Use sliding window for bg subtraction
            bg_algo="MEDIAN",
            dataset_labels=0,
            # diff_method="absdiff",
            diff_method="classical",
            pre_bg_transforms=create_realistic_pre_bg_transforms(),
            post_bg_transforms=create_post_bg_transforms(size=224),
            num_sequences_per_roi=self.DATAFACTOR,
            return_original=False,  # Also return original for comparison
        )

        dataset_real = OnlineBGSubtractionDataset(
            data_dirs=data_dirs,
            rois=rois,
            category="ovd",
            sequence_length=5,  # Fixed sequence length
            window_size=10,      # Use sliding window for bg subtraction
            bg_algo="MEDIAN",
            diff_method="classical",
            dataset_labels=1,
            # diff_method="classical",
            # pre_bg_transforms=create_realistic_pre_bg_transforms(),
            post_bg_transforms=create_post_valid_bg_transforms(size=224),
            num_sequences_per_roi=self.DATAFACTOR*3,
            return_original=False,  # Also return original for comparison
        )

        dataset_realsynt = OnlineBGSubtractionDataset(
            data_dirs=data_dirs,
            rois=rois,
            category="ovd",
            sequence_length=5,  # Fixed sequence length
            window_size=10,      # Use sliding window for bg subtraction
            bg_algo="MEDIAN",
            # diff_method="absdiff",
            diff_method="classical",
            # pre_bg_transforms=create_realistic_pre_bg_transforms(),
            post_bg_transforms=create_post_valid_bg_transforms(size=224),
            num_sequences_per_roi=self.DATAFACTOR,
            dataset_labels=0,
            return_original=False,  # Also return original for comparison
            fake_sequence_prob=1,  # High synthetic rate
            fake_sequence_strategies=["frame_repeat", "mixed"], # "shuffle", 
        )

        dataset_otherovd = OnlineBGSubtractionDataset(
            data_dirs=data_dirs,
            rois=rois,
            category="ovd2",
            sequence_length=5,  # Fixed sequence length
            window_size=10,      # Use sliding window for bg subtraction
            bg_algo="MEDIAN",
            dataset_labels=0,
            # diff_method="absdiff",
            diff_method="classical",
            pre_bg_transforms=create_realistic_pre_bg_transforms(),
            post_bg_transforms=create_post_bg_transforms(size=224),
            num_sequences_per_roi=self.DATAFACTOR,
            return_original=False,  # Also return original for comparison
        )

        self.train_dataset = ConcatDataset([dataset_otherROIs, dataset_real, dataset_realsynt, dataset_otherovd]) # change the *2

        data_dir_val = (Path(self.split_path) / "val.txt").open().read().splitlines()
        data_dirs_val = [f"{self.data_path}/origins/{d}/" for d in data_dir_val if Path(f"{self.data_path}/origins/{d}/img_0010.jpg").exists()]        
        # data_dirs = 
        val_datasets = [
            OnlineBGSubtractionDataset(
                data_dirs=data_dirs_val,
                rois=rois,
                category="ovd",
                sequence_length=5,  # Fixed sequence length
                window_size=10,      # Use sliding window for bg subtraction
                bg_algo="MEDIAN",
                diff_method="classical", ##### should be the same as above ...
                dataset_labels=1,
                # diff_method="classical",
                # pre_bg_transforms=create_realistic_pre_bg_transforms(),
                post_bg_transforms=create_post_valid_bg_transforms(size=224),
                return_all_sequences=True,
                sequence_stride=5, # stride se fait en comtant la seq totale utilisée pour le bgsub
                return_original=False,  # Also return original for comparison
            ),
            OnlineBGSubtractionDataset(
                data_dirs=data_dirs_val,
                rois=rois,
                category="other",
                sequence_length=5,  # Fixed sequence length
                window_size=10,      # Use sliding window for bg subtraction
                bg_algo="MEDIAN",
                dataset_labels=0,
                # diff_method="absdiff",
                diff_method="classical",
                pre_bg_transforms=create_realistic_pre_bg_transforms(),
                post_bg_transforms=create_post_bg_transforms(size=224),
                num_sequences_per_roi=self.DATAFACTOR,
                return_original=False,  # Also return original for comparison
            ),
            OnlineBGSubtractionDataset(
                data_dirs=data_dirs_val,
                rois=rois,
                category="ovd",
                sequence_length=5,  # Fixed sequence length
                window_size=10,      # Use sliding window for bg subtraction
                bg_algo="MEDIAN",
                # diff_method="absdiff",
                diff_method="classical",
                # pre_bg_transforms=create_realistic_pre_bg_transforms(),
                post_bg_transforms=create_post_valid_bg_transforms(size=224),
                num_sequences_per_roi=self.DATAFACTOR,
                dataset_labels=0,
                return_original=False,  # Also return original for comparison
                fake_sequence_prob=1,  # High synthetic rate
                fake_sequence_strategies=["frame_repeat", "mixed"], # "shuffle", 
            ),
            OnlineBGSubtractionDataset(
                data_dirs=data_dirs_val,
                rois=rois,
                category="ovd2",
                sequence_length=5,  # Fixed sequence length
                window_size=10,      # Use sliding window for bg subtraction
                bg_algo="MEDIAN",
                dataset_labels=0,
                # diff_method="absdiff",
                diff_method="classical",
                pre_bg_transforms=create_realistic_pre_bg_transforms(),
                post_bg_transforms=create_post_bg_transforms(size=224),
                num_sequences_per_roi=self.DATAFACTOR,
                return_original=False,  # Also return original for comparison
            )
        ]
        
        self.val_dataset = ConcatDataset(val_datasets)
  

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=lambda batch: {
                    'video_tensor': torch.stack([item['sequence'] for item in batch]),
                    "label": torch.tensor([item['label'] for item in batch]),
                    # 'original': torch.stack([item['original'] for item in batch]) if batch[0].get('original') is not None else None,
                    #'metadata': [{'sequence_idx': item['sequence_idx'], 'roi_idx': item['roi_idx'], 
                    #             'variant_idx': item['variant_idx']} for item in batch]
                }
            )
    
    def val_dataloader(self):
        return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=lambda batch: {
                    'video_tensor': torch.stack([item['sequence'] for item in batch]),
                    "label": torch.tensor([item['label'] for item in batch]),
                    # 'original': torch.stack([item['original'] for item in batch]) if batch[0].get('original') is not None else None,
                    #'metadata': [{'sequence_idx': item['sequence_idx'], 'roi_idx': item['roi_idx'], 
                    #             'variant_idx': item['variant_idx']} for item in batch]
                }
            )
    

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
        self.quickdraw_masks = torch.load("sequenceclassif/quickdraw_subsamples_more.pt").float()
        self.quickdraw_masks = self.quickdraw_masks.reshape(-1, 28, 28) 

    def apply_transform(
        self, image: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
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
