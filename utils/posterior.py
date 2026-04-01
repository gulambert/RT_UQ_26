import math
import torch
from torch import Tensor

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from gpytorch.distributions import MultivariateNormal

from k_means_constrained import KMeansConstrained

from typing import Tuple, List


def batched_posterior_derivative_joint_fantasize(model, Xs, return_full: bool = True):
    r"""
    Computes the joint gradient posterior for a (fantasized) GP model.
    
    Args:
        model: A fantasized GP model with the following assumptions:
               - model.train_inputs[0] has shape (n_fant, B, n_train, d) OR can be
                 unbatched (n_train, d) or candidate-batched (B, n_train, d). In these cases,
                 the inputs are normalized to (n_fant, B, n_train, d).
               - model.train_targets has shape (n_fant, B, n_train)
               - model.likelihood.noise has shape (B, n_train)
               - model.mean_module.constant is a scalar.
               - The kernel is ARD Matern 5/2 with parameters:
                     lengthscale: tensor of shape (d,)
                     outputscale: scalar
        Xs: Test (global) input points of shape (N, d)
        return_full: If False, return only the flattened gradient mean without computing the covariance.
    
    Returns:
        If return_full is True (default): A gpytorch.distribution.MultivariateNormal with
          - mean of shape (n_fant, B, N*d)
          - covariance_matrix of shape (n_fant, B, N*d, N*d)
        Otherwise: Tensor of shape (n_fant, B, N*d) containing the flattened gradient mean.

    Mainly based on GPytorch Matern52Grad Kernel implementation.
    """
    device = Xs.device
    dtype = Xs.dtype
    N, d = Xs.shape

    train_in = model.train_inputs[0]
    if train_in.ndim == 2:
        train_in = train_in.unsqueeze(0).unsqueeze(0)  # (1, 1, n_train, d)
    elif train_in.ndim == 3:
        if train_in.shape[0] > 1:
            train_in = train_in.unsqueeze(1)  # (n_fant, 1, n_train, d)
        else:
            train_in = train_in.unsqueeze(0)  # (1, B, n_train, d)
    elif train_in.ndim == 4:
        pass
    else:
        raise ValueError("Unsupported number of dimensions for model.train_inputs[0].")
    
    y_train = model.train_targets
    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(0).unsqueeze(0)  # (1, 1, n_train)
    elif y_train.ndim == 2:
        if y_train.shape[0] > 1:
            y_train = y_train.unsqueeze(1)  # (n_fant, 1, n_train)
        else:
            y_train = y_train.unsqueeze(0)  # (1, B, n_train)
    elif y_train.ndim == 3:
        pass
    else:
        raise ValueError("Unsupported number of dimensions for model.train_targets.")
    
    noise = model.likelihood.noise
    if noise.ndim == 0:
        noise = noise.expand(train_in.shape[1], train_in.shape[2])
    elif noise.ndim == 1:
        noise = noise.unsqueeze(0).expand(train_in.shape[1], train_in.shape[2])
    elif noise.ndim == 2:
        pass
    else:
        raise ValueError("Unsupported number of dimensions for model.likelihood.noise.")
    
    n_fant, B, n_train, d_train = train_in.shape
    assert d == d_train, "Dimension mismatch between test and training inputs."

    m_val = model.mean_module.constant.detach()  # assumed constant mean
    lengthscales = model.covar_module.base_kernel.lengthscale.squeeze()  # shape (d,)
    outputscale = model.covar_module.outputscale.item()  # scalar
    
    K_train = model.covar_module(train_in, train_in).to_dense()
    eye_n = torch.eye(n_train, dtype=dtype, device=device)
    noise = noise.clamp_min(1e-6)
    K_train = K_train + noise.unsqueeze(0).unsqueeze(-1) * eye_n
    jitter = 1e-6
    L = torch.linalg.cholesky(K_train + jitter * eye_n)# shape (n_fant, B, n_train, n_train)
    
    diff_train = y_train - m_val  # shape (n_fant, B, n_train)
    M_diff = torch.cholesky_solve(diff_train.unsqueeze(-1), L).squeeze(-1)  # shape (n_fant, B, n_train)
    
    Xs_exp = Xs.unsqueeze(0).unsqueeze(0).unsqueeze(3)  # (1, 1, N, 1, d) to subtract training points
    X_train_exp = train_in.unsqueeze(2)  # (n_fant, B, 1, n_train, d)
    diff = Xs_exp - X_train_exp  # shape (n_fant, B, N, n_train, d)
    
    scaled_diff = diff / lengthscales.view(1, 1, 1, 1, d)  # (d,) -> (1,1,1,1,d)
    sqrt5 = torch.sqrt(torch.tensor(5.0, dtype=dtype, device=device))
    r = torch.norm(scaled_diff, dim=-1)  # shape (n_fant, B, N, n_train)
    
    factor = -5 * outputscale / 3 * (1 + sqrt5 * r) * torch.exp(-sqrt5 * r)  # (n_fant, B, N, n_train)
    grad_K = factor.unsqueeze(-1) * (diff / (lengthscales**2).view(1, 1, 1, 1, d))
    
    grad_mean = torch.einsum('fbikd,fbk->fbid', grad_K, M_diff)  # shape (n_fant, B, N, d)
    grad_mean_flat = grad_mean.reshape(n_fant, B, N * d)  # shape (n_fant, B, N*d)

    if not return_full:
        return grad_mean_flat
    
    diff_tt = Xs.unsqueeze(1) - Xs.unsqueeze(0)  # differences between test points: (N, N, d)
    scaled_diff_tt = diff_tt / lengthscales.view(1, 1, d)  # (N, N, d)
    r_tt = torch.norm(scaled_diff_tt, dim=-1)  # (N, N)
    h_val_tt = (1 + sqrt5 * r_tt) * torch.exp(-sqrt5 * r_tt)  # (N, N)
    outer_tt = diff_tt.unsqueeze(-1) * (diff_tt / (lengthscales**2).view(1, 1, d)).unsqueeze(-2)  # (N, N, d, d)
    A = (5 * outputscale / 3) / (lengthscales**2)  # (d,)
    eye_d = torch.eye(d, dtype=dtype, device=device)
    exp_factor = 5 * torch.exp(-sqrt5 * r_tt)
    # H_prior: (N, N, d, d)
    H_prior = - A.view(1, 1, d, 1) * (exp_factor.unsqueeze(-1).unsqueeze(-1) * outer_tt -
                                       h_val_tt.unsqueeze(-1).unsqueeze(-1) * eye_d)
    H_prior_batched = H_prior.unsqueeze(0).unsqueeze(0).expand(n_fant, B, N, N, d, d) 


    L_exp = L.unsqueeze(2).expand(n_fant, B, N, n_train, n_train)
    B_sol = torch.cholesky_solve(grad_K, L_exp)  # shape (n_fant, B, N, n_train, d)
    
    cross_term = torch.einsum('fbikd,fbjke->fbijde', grad_K, B_sol)  # shape (n_fant, B, N, N, d, d)

    cov_grad = H_prior_batched - cross_term
    
    cov_grad = cov_grad.permute(0, 1, 2, 4, 3, 5).reshape(n_fant, B, N * d, N * d)
    cov_grad = 0.5 * (cov_grad + cov_grad.transpose(-1, -2))
    cov_grad = cov_grad + 1e-6 * torch.eye(N*d, device=device, dtype=dtype).expand(n_fant, B, N*d, N*d)

    return MultivariateNormal(grad_mean_flat, cov_grad)


def _resolve_chunk_size_points(d: int, device_type: str, requested: int = None) -> int:
    """Keep chunk_size*d bounded to avoid blowing up the covariance blocks."""
    base = 500 if device_type == "cuda" else 2000
    cap = max(1, base // max(1, d))
    if requested is None:
        return cap
    requested = int(max(1, requested))
    return min(requested, cap)

def batched_posterior_derivative_joint_fantasize_chunked_kmeans(
    model,
    Xs,
    chunk_size: int = 100,
    selection_alg: str = "kmeans-equal",
    verbose: bool = False,
    return_indices: bool = False,
):
    """
    Memory-efficient adaptation of batched_posterior_derivative_joint_fantasize.

    chunk_size s is from C = \ceil(N / s); N = Xs.shape[0]
    """
    device = Xs.device
    dtype = Xs.dtype
    N, d = Xs.shape

    chunk_size = _resolve_chunk_size_points(d, device.type, chunk_size) #Naive chunk size allocation, arbitrary
    n_chunks = max(1, (N + chunk_size - 1) // chunk_size)
    labels = None
    chunk_indices: List[torch.Tensor] = []
    Xs_np = Xs.detach().cpu().numpy()

    if selection_alg in ("kmeans-equal", "balanced"):
        if verbose:
            print("Performing balanced k-means clustering...")
        kmeans = KMeansConstrained(
            n_clusters=n_chunks,
            size_min=N // n_chunks,
            size_max=math.ceil(N / n_chunks),
            init="k-means++",
            max_iter=2000,
            tol=1e-5,
            n_jobs=-1,
        )
        kmeans.fit(Xs_np)
        labels = kmeans.labels_
    elif selection_alg in ("kmeans", "default"):

        if verbose:
            print("Performing classical k-means clustering...")
        kmeans = KMeans(
            n_clusters=n_chunks,
            init="k-means++",
        )
        labels = kmeans.fit_predict(Xs_np)
    elif selection_alg == "random":
        pass

    elif selection_alg == "pca-kmeans":
        pca_dim = min(d, 3)
        pca = PCA(n_components=pca_dim)
        Xs_reduced = pca.fit_transform(Xs_np)
        kmeans = KMeans(n_clusters=n_chunks, init="k-means++", max_iter=300)
        labels = kmeans.fit_predict(Xs_reduced)
    else:
        raise ValueError(
            "Unsupported selection algorithm. "
            "Use 'default', 'balanced', 'random', or 'pca-kmeans'."
        )
    train_in = model.train_inputs[0]
    if train_in.ndim == 2:
        train_in = train_in.unsqueeze(0).unsqueeze(0)  # (1, 1, n_train, d)
    elif train_in.ndim == 3:
        if train_in.shape[0] > 1:
            train_in = train_in.unsqueeze(1)  # (n_fant, 1, n_train, d)
        else:
            train_in = train_in.unsqueeze(0)  # (1, B, n_train, d)
    elif train_in.ndim == 4:
        pass
    else:
        raise ValueError("Unsupported dimensions for model.train_inputs[0].")
    
    y_train = model.train_targets
    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(0).unsqueeze(0)  # (1, 1, n_train)
    elif y_train.ndim == 2:
        if y_train.shape[0] > 1:
            y_train = y_train.unsqueeze(1)  # (n_fant, 1, n_train)
        else:
            y_train = y_train.unsqueeze(0)  # (1, B, n_train)
    elif y_train.ndim == 3:
        pass
    else:
        raise ValueError("Unsupported dimensions for model.train_targets.")
    
    noise = model.likelihood.noise
    if noise.ndim == 0:
        noise = noise.expand(train_in.shape[1], train_in.shape[2])
    elif noise.ndim == 1:
        noise = noise.unsqueeze(0).expand(train_in.shape[1], train_in.shape[2])
    elif noise.ndim == 2:
        pass
    else:
        raise ValueError("Unsupported dimensions for model.likelihood.noise.")
    
    n_fant, B, n_train, d_train = train_in.shape
    assert d == d_train, "Dimension mismatch between test and training inputs."

    m_val = model.mean_module.constant.detach()
    lengthscales = model.covar_module.base_kernel.lengthscale.squeeze()
    outputscale = model.covar_module.outputscale.item()
    
    K_train = model.covar_module(train_in, train_in).to_dense()
    eye_n = torch.eye(n_train, dtype=dtype, device=device)
    noise = noise.clamp_min(1e-6)
    K_train = K_train + noise.unsqueeze(0).unsqueeze(-1) * eye_n
    jitter = 1e-6
    L = torch.linalg.cholesky(K_train + jitter * eye_n)
    
    diff_train = y_train - m_val
    M_diff = torch.cholesky_solve(diff_train.unsqueeze(-1), L).squeeze(-1)
    
    sqrt5 = torch.sqrt(torch.tensor(5.0, dtype=dtype, device=device))
    
    A = (5 * outputscale / 3) / (lengthscales**2)
    eye_d = torch.eye(d, dtype=dtype, device=device)
    
    chunk_results = []
    
    for i in range(n_chunks):
        chunk_idx = None
        if selection_alg in ("kmeans-equal", "balanced", "kmeans", "default", "pca-kmeans"):
            cluster_mask = torch.as_tensor(labels == i, device=device, dtype=torch.bool)
            chunk_N = int(cluster_mask.sum().item())
            if chunk_N == 0:
                continue
            Xs_chunk = Xs[cluster_mask]
            if return_indices:
                chunk_idx = torch.nonzero(cluster_mask, as_tuple=False).view(-1)
        elif selection_alg == "random":
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, N)
            Xs_chunk = Xs[start_idx:end_idx]
            chunk_N = end_idx - start_idx
            if return_indices:
                chunk_idx = torch.arange(start_idx, end_idx, device=device, dtype=torch.long)
        else:
            raise ValueError("Tu as mal sélectionné l'alg de sélection. Try again :)")
        
        Xs_exp = Xs_chunk.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        X_train_exp = train_in.unsqueeze(2)
        diff = Xs_exp - X_train_exp
        
        scaled_diff = diff / lengthscales.view(1, 1, 1, 1, d)
        r = torch.norm(scaled_diff, dim=-1)
        
        factor = -5 * outputscale / 3 * (1 + sqrt5 * r) * torch.exp(-sqrt5 * r)
        grad_K = factor.unsqueeze(-1) * (diff / (lengthscales**2).view(1, 1, 1, 1, d))
        
        grad_mean = torch.einsum('fbikd,fbk->fbid', grad_K, M_diff)
        grad_mean_flat = grad_mean.reshape(n_fant, B, chunk_N * d)
        
        diff_tt = Xs_chunk.unsqueeze(1) - Xs_chunk.unsqueeze(0)
        scaled_diff_tt = diff_tt / lengthscales.view(1, 1, d)
        r_tt = torch.norm(scaled_diff_tt, dim=-1)
        h_val_tt = (1 + sqrt5 * r_tt) * torch.exp(-sqrt5 * r_tt)
        outer_tt = diff_tt.unsqueeze(-1) * (diff_tt / (lengthscales**2).view(1, 1, d)).unsqueeze(-2)
        
        exp_factor = 5 * torch.exp(-sqrt5 * r_tt)
        H_prior = - A.view(1, 1, d, 1) * (exp_factor.unsqueeze(-1).unsqueeze(-1) * outer_tt -
                                           h_val_tt.unsqueeze(-1).unsqueeze(-1) * eye_d)
        H_prior_batched = H_prior.unsqueeze(0).unsqueeze(0).expand(n_fant, B, chunk_N, chunk_N, d, d)
        
        L_exp = L.unsqueeze(2).expand(n_fant, B, chunk_N, n_train, n_train)
        B_sol = torch.cholesky_solve(grad_K, L_exp)
        cross_term = torch.einsum('fbikd,fbjke->fbijde', grad_K, B_sol)
        
        cov_grad = H_prior_batched - cross_term
        cov_grad = cov_grad.permute(0, 1, 2, 4, 3, 5).reshape(n_fant, B, chunk_N * d, chunk_N * d)
        cov_grad = 0.5 * (cov_grad + cov_grad.transpose(-1, -2))
        eye_chunk = torch.eye(chunk_N * d, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        cov_grad = cov_grad + 1e-6 * eye_chunk.expand(n_fant, B, chunk_N * d, chunk_N * d)
        
        chunk_results.append((grad_mean_flat, cov_grad))

        if return_indices:
            if chunk_idx is None:
                raise RuntimeError("Missing cluster indices for chunked posterior.")
            chunk_indices.append(chunk_idx)

    if return_indices:
        return chunk_results, chunk_indices
    return chunk_results


def matern52_grad_first_arg(
    X1: Tensor,
    X2: Tensor,
    lengthscales: Tensor,
    outputscale: Tensor,
) -> Tensor:
    """
    grad[i,j,a] = d/dX1[i,a] k(X1[i], X2[j])
    Extract from batched_posterior_derivative_joint_fantasize.
    """
    diff = X1[:, None, :] - X2[None, :, :]
    d = X1.shape[-1]
    ls = lengthscales.view(1, 1, d)
    scaled_diff = diff / ls
    r = torch.linalg.norm(scaled_diff, dim=-1)
    sqrt5 = torch.sqrt(torch.tensor(5.0, dtype=X1.dtype, device=X1.device))
    factor = -(5.0 * outputscale / 3.0) * (1.0 + sqrt5 * r) * torch.exp(-sqrt5 * r)
    grad = factor.unsqueeze(-1) * diff / (lengthscales.view(1, 1, d) ** 2)
    return grad

def posterior_quadratic_form_variance(mu: Tensor, cov: Tensor) -> Tensor:
    """
    Var(Z^T Z) for Gaussian Z ~ N(mu, cov).
    """
    tr_sig2 = 2.0 * (cov ** 2).sum(dim=(-2, -1))
    sig_mu = torch.einsum("...ij,...j->...i", cov, mu)
    mu_sig_mu = torch.einsum("...i,...i->...", mu, sig_mu)
    return tr_sig2 + 4.0 * mu_sig_mu