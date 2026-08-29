# evaluation/grouping.py
"""Fraud type grouping and aggregation utilities."""

import numpy as np
from typing import Dict, List, Tuple

# ══════════════════════════════════════════════════════════════
# Group definitions
# ══════════════════════════════════════════════════════════════

FRAUD_GROUPS = {
    'static_midvholo': [
        'copy_without_holo', 'pseudo_holo_copy', 'photo_holo_copy'
    ],
    'static_midvdynattack': [
        'no_holo', 'laser', 'plastified_led', 'plastified_lowreflect', 'plastified_noholo'
    ],
    'dynamic_midvdynattack': [
        'holo_star_world', 'holo_completemask', 'leaf_holo', 'plain_holo', 'double_sticker'
    ],
    'swap_midvdynattack': [
        'swap', 'swap_three'
    ],
    'photo_replacement': [
        'photo_replacement'
    ],
    'midvdynattack': [
        'no_holo', 'laser', 'plastified_led', 'plastified_lowreflect', 'plastified_noholo', 'holo_star_world', 'holo_completemask', 'leaf_holo', 'plain_holo', 'double_sticker', 'swap', 'swap_three'
    ],
}

# Reverse mapping: fraud_base -> group_name
FRAUD_TO_GROUP = {
    fraud: group 
    for group, frauds in FRAUD_GROUPS.items() 
    for fraud in frauds
}

FRAUD_TO_GROUPS: Dict[str, List[str]] = {}
for group, frauds in FRAUD_GROUPS.items():
    for fraud in frauds:
        FRAUD_TO_GROUPS.setdefault(fraud, []).append(group)


# ══════════════════════════════════════════════════════════════
# Merging _ID and _passport variants
# ══════════════════════════════════════════════════════════════

def get_fraud_base(name: str) -> str:
    """Strip _ID or _passport suffix to get base fraud type.
    
    Examples:
        'photo_holo_copy_ID' -> 'photo_holo_copy'
        'swap_passport' -> 'swap'
        'origins' -> 'origins'
    """
    if name == "origins":
        return name
    for suffix in ("_ID", "_passport"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def merge_id_passport_variants(
    all_scores: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Merge _ID and _passport variants into single fraud types.
    
    Parameters
    ----------
    all_scores : mapping name -> {signal -> 1D array}
                 e.g., {'photo_holo_copy_ID': {'recon_cosine': [...], ...}, ...}
    
    Returns
    -------
    merged : mapping base_name -> {signal -> concatenated array}
             e.g., {'photo_holo_copy': {'recon_cosine': [...], ...}, ...}
    """
    merged: Dict[str, Dict[str, List[np.ndarray]]] = {}
    
    for name, signals in all_scores.items():
        base = get_fraud_base(name)
        
        if base not in merged:
            merged[base] = {k: [] for k in signals.keys()}
        
        for signal, values in signals.items():
            merged[base][signal].append(values)
    
    # Concatenate arrays
    result = {}
    for base, signals in merged.items():
        result[base] = {
            signal: np.concatenate(arrays) 
            for signal, arrays in signals.items()
        }
    
    return result


# ══════════════════════════════════════════════════════════════
# Aggregating to group level
# ══════════════════════════════════════════════════════════════

def aggregate_to_groups_old(
    merged_scores: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate merged fraud types into groups.
    
    Parameters
    ----------
    merged_scores : output from merge_id_passport_variants()
    
    Returns
    -------
    grouped : mapping group_name -> {signal -> concatenated array}
    """
    grouped: Dict[str, Dict[str, List[np.ndarray]]] = {}
    
    for fraud_base, signals in merged_scores.items():
        if fraud_base == "origins":
            # Keep origins separate
            grouped["origins"] = {k: [v] for k, v in signals.items()}
            continue
        
        group_name = FRAUD_TO_GROUP.get(fraud_base)
        if group_name is None:
            print(f"  ⚠ Unknown fraud type '{fraud_base}' — skipping")
            continue
        
        if group_name not in grouped:
            grouped[group_name] = {k: [] for k in signals.keys()}
        
        for signal, values in signals.items():
            grouped[group_name][signal].append(values)
    
    # Concatenate
    result = {}
    for group_name, signals in grouped.items():
        result[group_name] = {
            signal: np.concatenate(arrays) 
            for signal, arrays in signals.items()
        }
    
    return result

def aggregate_to_groups(
    merged_scores: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate merged fraud types into groups.

    A fraud can belong to MULTIPLE groups (e.g. 'no_holo' is in both
    'static_midvdynattack' and 'midvdynattack'). Each group is computed
    independently by iterating forward over FRAUD_GROUPS, so overlapping
    memberships are correctly handled.
    """
    # Always carry origins through
    result: Dict[str, Dict[str, np.ndarray]] = {}
    if "origins" in merged_scores:
        result["origins"] = merged_scores["origins"]

    for group_name, fraud_list in FRAUD_GROUPS.items():
        buckets: Dict[str, List[np.ndarray]] = {}

        for fraud_base in fraud_list:
            if fraud_base not in merged_scores:
                continue                          # variant absent in this split
            signals = merged_scores[fraud_base]
            for signal, values in signals.items():
                buckets.setdefault(signal, []).append(values)

        if not buckets:
            print(f"  ⚠ Group '{group_name}' has no matching frauds — skipping")
            continue

        result[group_name] = {
            signal: np.concatenate(arrays)
            for signal, arrays in buckets.items()
        }

    return result


# ══════════════════════════════════════════════════════════════
# Full aggregation pipeline
# ══════════════════════════════════════════════════════════════

def build_all_aggregation_levels(
    all_scores: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],  # original (with _ID/_passport)
    Dict[str, Dict[str, np.ndarray]],  # merged (base fraud types)
    Dict[str, Dict[str, np.ndarray]],  # grouped (attack categories)
]:
    """Build all three aggregation levels.
    
    Returns
    -------
    (original_scores, merged_scores, grouped_scores)
    """
    merged = merge_id_passport_variants(all_scores)
    grouped = aggregate_to_groups(merged)
    
    return all_scores, merged, grouped


def get_group_composition_old(
    merged_scores: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, int]]:
    """Get sample counts per fraud type within each group.
    
    Returns
    -------
    composition : {group_name: {fraud_base: n_samples}}
    """
    composition = {}
    
    for fraud_base, signals in merged_scores.items():
        if fraud_base == "origins":
            continue
        
        group_name = FRAUD_TO_GROUP.get(fraud_base)
        if group_name is None:
            continue
        
        if group_name not in composition:
            composition[group_name] = {}
        
        n_samples = len(next(iter(signals.values())))
        composition[group_name][fraud_base] = n_samples
    
    return composition


def get_group_composition(
    merged_scores: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, int]]:
    """Get sample counts per fraud type within each group.

    Uses FRAUD_GROUPS directly so overlapping groups are all populated.
    """
    composition: Dict[str, Dict[str, int]] = {}

    for group_name, fraud_list in FRAUD_GROUPS.items():
        composition[group_name] = {}
        for fraud_base in fraud_list:
            if fraud_base not in merged_scores or fraud_base == "origins":
                continue
            signals = merged_scores[fraud_base]
            n = len(next(iter(signals.values())))
            composition[group_name][fraud_base] = n

    return composition