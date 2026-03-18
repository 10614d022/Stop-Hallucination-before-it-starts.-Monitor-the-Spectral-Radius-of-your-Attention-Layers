import torch


def power_iteration_top_eig(mat: torch.Tensor, n_iter: int = 20, eps: float = 1e-12) -> torch.Tensor:
    v = torch.randn(mat.size(0), device=mat.device, dtype=mat.dtype)
    v = v / (v.norm() + eps)
    for _ in range(n_iter):
        v = mat @ v
        v = v / (v.norm() + eps)
    return torch.dot(v, mat @ v)


def compute_token_level_rho(attn_2d: torch.Tensor) -> float:
    """
    attn_2d: [T, T]
    Implements C = XX^T / T with row-centering.
    """
    T = attn_2d.size(0)
    X = attn_2d - attn_2d.mean(dim=-1, keepdim=True)
    C = (X @ X.transpose(0, 1)) / max(T, 1)
    C = 0.5 * (C + C.transpose(0, 1))
    return float(power_iteration_top_eig(C).item())


def compute_entropy(attn_2d: torch.Tensor, eps: float = 1e-12) -> float:
    p = attn_2d.clamp_min(eps)
    row_entropy = -(p * p.log()).sum(dim=-1)
    return float(row_entropy.mean().item())
