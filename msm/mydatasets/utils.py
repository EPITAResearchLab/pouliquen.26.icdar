"""Dataset utilities including masking functions."""
import torch


def seq_mask(
    embeddings: torch.Tensor,
    len_hide: int = 5,
    num_masks: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = embeddings.shape
    masked = embeddings.clone()
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=embeddings.device)
    
    for _ in range(num_masks):
        starts = torch.randint(0, seq_len - len_hide + 1, (batch_size,))
        for b in range(batch_size):
            mask[b, starts[b]:starts[b] + len_hide] = True
    
    masked[mask] = 0
    return masked, mask


def bert_mask(
    embeddings: torch.Tensor,
    mask_prob: float = 0.8,
    noise_std: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = embeddings.clone()
    rand = torch.rand(embeddings.shape[:-1], device=embeddings.device)
    mask = rand < mask_prob
    
    rand_strategy = torch.rand(embeddings.shape[:-1], device=embeddings.device)
    
    # 80% zero out
    zero_mask = mask & (rand_strategy < 0.8)
    masked[zero_mask] = 0
    
    # 10% add noise
    noise_mask = mask & (rand_strategy >= 0.8) & (rand_strategy < 0.9)
    noise = torch.randn_like(embeddings) * noise_std
    masked[noise_mask] += noise[noise_mask]
    
    return masked, mask
