from typing import Sequence, Optional, Dict, Any
import torch
from torch import Tensor
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor
from scipy.stats import beta, triang
DistSpec = Dict[str, Any]

def transform_unit_to_marginals(
    U: Tensor,
    dists: Sequence[DistSpec],
    *,
    eps: float = 1e-10,
) -> Tensor:
    """
    Transform U in [0,1]^d to independent marginals described by `dists`.
     """
    d_total = sum(
        1 if spec["type"].lower() != "mvn" else torch.as_tensor(spec["mu"]).numel()
        for spec in dists
    )
    if U.shape[-1] != d_total:
        raise ValueError(f"U.shape[-1]={U.shape[-1]} but the summed dimension of dists is {d_total}")

    U = U.clamp(min=eps, max=1.0 - eps)
    X = torch.empty_like(U)

    j = 0 
    for spec in dists:
        t = spec["type"].lower()

        if t == "mvn":
            mu = torch.as_tensor(spec["mu"])
            cov = torch.as_tensor(spec["cov"])
            k = mu.numel()
            u_block = U[..., j : j + k]
            X[..., j : j + k] = _ppf_mvn(u_block, mu=mu, cov=cov)
            j += k
            continue

        ui = U[..., j]
        if t in ("exponential", "exp"):
            X[..., j] = _ppf_exponential(ui, rate=float(spec["rate"]))
        elif t in ("gumbel", "gumbel_r"):
            X[..., j] = _ppf_gumbel(
                ui,
                loc=float(spec.get("loc", 0.0)),
                scale=float(spec.get("scale", 1.0)),
            )
        elif t == "beta":
            X[..., j] = _ppf_beta(ui, a=float(spec["a"]), b=float(spec["b"]))
        elif t == "beta_scale":
            X[..., j] = _ppf_beta(ui, a=float(spec["a"]), b=float(spec["b"]), low = float(spec["low"]), high = float(spec["high"]))
        elif t in ("uniform", "unif"):
            X[..., j] = _uniform_untransform(ui, a=float(spec["a"]), b=float(spec["b"]))
        elif t in ("normal", "norm"):
            X[..., j] = _ppf_norm(ui,mu=float(spec["mu"]), sigma=float(spec["sigma"]))
        elif t in ("triangular", "triang"):
            X[..., j] = _ppf_triangular(ui, left=float(spec["left"]), center=float(spec["center"]),right=float(spec["right"]),)
        elif t in ("lognormal", "lognorm"):
            X[..., j] = _ppf_lognormal(ui,mu=float(spec["mu"]),sigma=float(spec["sigma"]),)
        else:
            raise ValueError(f"Unknown distribution type: {spec['type']}")
        j += 1

    return X


def _ppf_exponential(u: Tensor, rate: float) -> Tensor:
    return -torch.log1p(-u) / rate

def _ppf_gumbel(u: Tensor, loc: float, scale: float) -> Tensor:
    return loc - scale * torch.log(-torch.log(u))

def _ppf_beta(u: Tensor, a: float, b: float, low: float = 0, high:float=1) -> Tensor:
    u_np = u.detach().cpu().numpy()
    x_np = low + beta.ppf(u_np, a=a, b=b) * (high - low) # B(a,b)sur [low, high] 
    return torch.as_tensor(x_np, device=u.device, dtype=u.dtype)

def _ppf_norm(u:Tensor, mu:float, sigma : float) -> Tensor:
    norm_dist = torch.distributions.normal.Normal(loc = mu, scale = sigma)
    return norm_dist.icdf(u)
def _ppf_mvn(u_block: Tensor, mu: Tensor, cov: Tensor) -> Tensor:
    """
    Map a block of uniforms u_block in (0,1)^(..., k) to MVN(mu, cov)
    using a Gaussian transform:
      z = Phi^{-1}(u), x = mu + z @ L^T, where L = chol(cov).
    """
    mu = mu.to(device=u_block.device, dtype=u_block.dtype)
    cov = cov.to(device=u_block.device, dtype=u_block.dtype)

    k = mu.numel()
    if u_block.shape[-1] != k:
        raise ValueError(f"u_block.shape[-1]={u_block.shape[-1]} but mu has {k} elements :(.")
    z = torch.distributions.Normal(0.0, 1.0).icdf(u_block) # (..., k)
    L = torch.linalg.cholesky(cov)# (k, k)

    return z @ L.transpose(-1, -2) + mu  #(..., k)

def _ppf_triangular(u: Tensor, left: float, center: float, right: float) -> Tensor:
    c = (center-left)/(right-left)
    u_np = u.detach().cpu().numpy()
    x_np = triang.ppf(u_np, c=c, loc=left, scale=right - left)
    return torch.as_tensor(x_np, device=u.device, dtype=u.dtype)
def _ppf_lognormal(u: Tensor, mu: float, sigma: float) -> Tensor:
    """
    Underlying normal has mean mu and std sigma, then:
        X = exp(mu + sigma * Phi^{-1}(u))
    """
    norm_dist = torch.distributions.Normal(loc=0.0, scale=1.0)
    return torch.exp(mu+sigma*norm_dist.icdf(u))
def _uniform_untransform(u: Tensor, a: float, b:float) -> Tensor:
    return a + u * (b-a)

def draw_samples(
    dists: Sequence[DistSpec],
    n: int,
    q: int = 1,
    seed: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    r"""
    Draw qMC Sobol samples from independent marginals.
    Args:
        dists: list of per-dimension distribution specs, length d. Each spec is:
            - {"type": "beta", "a": ..., "b": ...}
            - {"type": "exponential", "rate": ...}
            - {"type": "gumbel", "loc": ..., "scale": ...}
            - {"type" : "uniform", "a": ..., "b":...}
            - {"type : "mvn", "mu": ..., "cov":...}
        n: number of q-batches.
        q: number of points per q-batch.
        seed: Sobol scrambling seed.
        dtype, device: torch dtype/device for the output.

    Returns:
        Tensor of shape `n x batch_shape x q x d`.
    """
    d = sum(
        1 if dist["type"].lower() != "mvn" else torch.as_tensor(dist["mu"]).numel()
        for dist in dists
    )

    u = draw_sobol_samples(
        bounds=torch.tensor([d * [0.0], d * [1.0]]),
        n=n,
        q=q,
        seed=seed,
    ).to(device=device, dtype=dtype)  # n x q x d

    return transform_unit_to_marginals(u, dists)