import numpy as np
import torch
from torch import Tensor


from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.utils.gpytorch_modules import get_matern_kernel_with_gamma_prior
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler


from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from utils.posterior import (
    batched_posterior_derivative_joint_fantasize,
    batched_posterior_derivative_joint_fantasize_chunked_kmeans,
)

from typing import Optional, List, Dict, Union, Any, Tuple
ArrayOrTensor = Union[np.ndarray, torch.Tensor]

torch.set_default_dtype(torch.float64)

from utils.posterior import matern52_grad_first_arg, posterior_quadratic_form_variance


class MinMaxNormalizer:
    """Min-max normalizer using torch tensors: X -> (X - min)/(max-min)"""
    def __init__(self, device: Union[str, torch.device] = 'cpu', dtype: torch.dtype = torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype
        self.min_: Optional[torch.Tensor] = None
        self.max_: Optional[torch.Tensor] = None
        self._rng: Optional[torch.Tensor] = None

    def _ensure_tensor(self, X: ArrayOrTensor) -> torch.Tensor:
        """Convert input to torch tensor with correct device and dtype."""
        if isinstance(X, np.ndarray):
            return torch.as_tensor(X, dtype=self.dtype, device=self.device)
        elif isinstance(X, torch.Tensor):
            return X.to(dtype=self.dtype, device=self.device)
        else:
            raise TypeError("Unsupported input type; expected numpy or torch tensor")

    def fit(self, X: ArrayOrTensor):
        """Fit the normalizer on input data."""
        X = self._ensure_tensor(X)
        self.min_ = X.min(dim=0, keepdim=True)[0]
        self.max_ = X.max(dim=0, keepdim=True)[0]
        rng = (self.max_ - self.min_)
        rng[rng == 0] = 1.0
        self._rng = rng
        return self

    def transform(self, X: ArrayOrTensor) -> torch.Tensor:
        """Transform data using fitted normalizer."""
        X = self._ensure_tensor(X)
        return (X - self.min_) / self._rng

    def transform_pca(self, X: ArrayOrTensor, M: int) -> torch.Tensor:
        """Transform data using last M dimensions of fitted normalizer."""
        X = self._ensure_tensor(X)
        return (X - self.min_[:, -M:]) / self._rng[:, -M:]

    def inverse_transform(self, Xn: ArrayOrTensor) -> torch.Tensor:
        """Inverse transform normalized data back to original scale."""
        Xn = self._ensure_tensor(Xn)
        return Xn * self._rng + self.min_


class Standardizer:
    """Per-dimension z-score: Y -> (Y - mean)/std; using torch tensors."""
    def __init__(self, eps: float = 1e-12, device: Union[str, torch.device] = 'cpu', dtype: torch.dtype = torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype
        self.mean_: Optional[torch.Tensor] = None
        self.std_: Optional[torch.Tensor] = None
        self.eps = eps

    def _ensure_tensor(self, Y: ArrayOrTensor) -> torch.Tensor:
        """Convert input to torch tensor with correct device and dtype."""
        if isinstance(Y, np.ndarray):
            return torch.as_tensor(Y, dtype=self.dtype, device=self.device)
        elif isinstance(Y, torch.Tensor):
            return Y.to(dtype=self.dtype, device=self.device)
        else:
            raise TypeError("Unsupported input type; expected numpy or torch tensor")

    def fit(self, Y: ArrayOrTensor):
        """Fit the standardizer on input data."""
        Y = self._ensure_tensor(Y)
        self.mean_ = Y.mean(dim=0, keepdim=True)
        std = Y.std(dim=0, keepdim=True, unbiased=False)
        std[std < self.eps] = 1.0
        self.std_ = std
        return self

    def transform(self, Y: ArrayOrTensor) -> torch.Tensor:
        """Transform data using fitted standardizer."""
        Y = self._ensure_tensor(Y)
        return (Y - self.mean_) / self.std_

    def inverse_transform(self, Yn: ArrayOrTensor) -> torch.Tensor:
        """Inverse transform standardized data back to original scale."""
        Yn = self._ensure_tensor(Yn)
        return Yn * self.std_ + self.mean_


class GP:
    """
    Gaussian Process class supporting:
      - 'independent' : single-output SingleTaskGP
      - 'mo_indep'    : multi-output via ModelListGP i.e a collection independent GPs

    noise=False -> y = f(x)  : Interpolation
    noise=True -> y = f(x) + e : e ~ N(0, sigma2) with sigma2 learned from data

    no_std_norm = True -> no MinMax / Scaling     /!\ Odd things can occur /!\   

    """

    def __init__(
        self,
        model_type: str = "independent",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        no_std_norm: bool = False,
        noise: bool = False,                  
    ):
        assert model_type in ("independent", "mo_indep"), \
            "model_type must be 'independent', 'mo_indep'."
        self.model_type = model_type
        self.device = torch.device(device)
        self.dtype = dtype
        self.no_std_norm = no_std_norm
        self.noise = noise             

        if not self.no_std_norm:
            self._x_normalizer = MinMaxNormalizer(device=device, dtype=dtype)
            self._y_standardizer = Standardizer(device=device, dtype=dtype)
        else:
            self._x_normalizer = None
            self._y_standardizer = None

        self.model= None
        self.models_list = None
        self.model_indep = None

        self.X_train = None
        self.Y_train = None
        self.input_dim = None
        self.output_dim = None

    def _ensure_tensor(self, X: ArrayOrTensor) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            return torch.as_tensor(X, dtype=self.dtype, device=self.device)
        if isinstance(X, torch.Tensor):
            return X.to(dtype=self.dtype, device=self.device)
        raise TypeError("Expected numpy array or torch.Tensor.")

    def _make_train_Yvar(self, Y: torch.Tensor) -> Optional[torch.Tensor]:
        if self.noise:
            return None
        return 1e-5 * torch.ones_like(Y)

    def _get_learned_noise(self, gp_model) -> Optional[torch.Tensor]:
        """
        Returns the learned noise variance as a scalar tensor,
        or None if noise=False (fixed noise, not meaningful for prediction).
        Used to add observation noise to predictive variance when noise=True.
        """
        if not self.noise:
            return None
        try:
            return gp_model.likelihood.noise.squeeze()
        except AttributeError:
            return None

    def _unstandardize_mvn(self, mvn_std, jitter: float = 1e-6):
        if self.no_std_norm:
            return mvn_std

        if (self._y_standardizer is None
                or self._y_standardizer.mean_ is None
                or self._y_standardizer.std_ is None):
            raise RuntimeError("Standardizer parameters missing.")

        std_y = self._y_standardizer.std_.view(-1)
        mean_y = self._y_standardizer.mean_.view(-1)
        R = std_y.shape[0]

        dev, dt = mvn_std.mean.device, mvn_std.mean.dtype
        std_y = std_y.to(dev, dt)
        mean_y = mean_y.to(dev, dt)

        raw_mean = mvn_std.mean
        raw_cov = mvn_std.covariance_matrix

        if R == 1 or raw_mean.shape[-1] != R:
            s, m = std_y[0], mean_y[0]
            mean_orig = raw_mean * s + m
            cov_orig = raw_cov  * (s ** 2)
            NR = raw_mean.shape[-1]
        else:
            N = raw_mean.shape[-2]
            mean_orig_2d = raw_mean * std_y + mean_y
            mean_orig = mean_orig_2d.reshape(*mean_orig_2d.shape[:-2], N * R)
            scale_vec = std_y.repeat(N)
            extra_dims = raw_cov.dim() - 2
            for _ in range(extra_dims):
                scale_vec = scale_vec.unsqueeze(0)
            cov_scale = scale_vec.unsqueeze(-1) * scale_vec.unsqueeze(-2)
            cov_orig = raw_cov * cov_scale
            NR = N * R

        eye = torch.eye(NR, device=dev, dtype=dt)
        for _ in range(cov_orig.dim() - 2):
            eye = eye.unsqueeze(0)
        cov_orig = cov_orig + jitter * eye

        return MultivariateNormal(mean_orig, cov_orig)

    def fit(
        self,
        x_array: ArrayOrTensor,
        y_array: ArrayOrTensor,
        verbose: bool = True,
        state_dict: Optional[Dict[str, Any]] = None,
    ):
        x_tensor = self._ensure_tensor(x_array)
        y_tensor = self._ensure_tensor(y_array)

        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)
        if x_tensor.dim() != 2 or y_tensor.dim() != 2:
            raise ValueError("x_array and y_array must be 2-D.")
        if x_tensor.shape[0] != y_tensor.shape[0]:
            raise ValueError("x_array and y_array must share the same number of samples.")

        self.input_dim = x_tensor.shape[1]
        self.output_dim = y_tensor.shape[1]

        if state_dict is not None:
            self.load_state_dict(state_dict)

            X_norm = x_tensor if self.no_std_norm else self._x_normalizer.transform(x_tensor)
            Y_norm = y_tensor if self.no_std_norm else self._y_standardizer.transform(y_tensor)
            self.X_train, self.Y_train = X_norm, Y_norm

            if self.model_type == "independent":
                restored = self._fit_independent_model(
                    X_norm, Y_norm, model_state=state_dict.get("model_state")
                )
            elif self.model_type == "mo_indep":
                restored = self._fit_independent_models(
                    X_norm, Y_norm, model_states=state_dict.get("model_states")
                )

            if verbose:
                origin = "restored" if restored else "refit"
                print(f"[GP.fit] {origin} from state_dict "
                      f"(model_type={self.model_type}, noise={self.noise})")
            return self

        if self.no_std_norm:
            X_norm, Y_norm = x_tensor, y_tensor
        else:
            self._x_normalizer.fit(x_tensor)
            self._y_standardizer.fit(y_tensor)
            X_norm = self._x_normalizer.transform(x_tensor)
            Y_norm = self._y_standardizer.transform(y_tensor)

        self.X_train, self.Y_train = X_norm, Y_norm

        if self.model_type == "independent":
            self._fit_independent_model(X_norm, Y_norm)
        elif self.model_type == "mo_indep":
            self._fit_independent_models(X_norm, Y_norm)

        if verbose:
            print(f"[GP.fit] trained from scratch: n={x_tensor.shape[0]}, "
                  f"model_type={self.model_type}, noise={self.noise}, "
                  f"no_std_norm={self.no_std_norm}")
        return self

    def _fit_independent_model(
        self,
        X_norm: torch.Tensor,
        Y_norm: torch.Tensor,
        model_state: Optional[Dict[str, Any]] = None,
    ) -> bool:

        covar = get_matern_kernel_with_gamma_prior(
                ard_num_dims=X_norm.shape[-1],
            )
        gp = SingleTaskGP(
            train_X=X_norm,
            train_Y=Y_norm,
            covar_module = covar,
            outcome_transform=None,
            train_Yvar=self._make_train_Yvar(Y_norm),
        ).to(self.device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        loaded = False
        if model_state is not None:
            try:
                gp.load_state_dict(model_state)
                loaded = True
            except Exception as exc:
                print(f"[GP.fit] Warning: model state failed: {exc}. Refitting.")

        if not loaded:
            fit_gpytorch_mll(mll)

        self.model = gp
        self.models_list= None
        self.model_indep = None
        return loaded

    def _fit_independent_models(
        self,
        X_norm: torch.Tensor,
        Y_norm: torch.Tensor,
        model_states: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        self.models_list = []
        all_restored = bool(model_states)
        covar = get_matern_kernel_with_gamma_prior(
                ard_num_dims=X_norm.shape[-1],
            )

        for r in range(self.output_dim):
            y_r = Y_norm[:, r:r+1]
            gp = SingleTaskGP(
                train_X=X_norm,
                train_Y=y_r,
                covar_module = covar,
                outcome_transform=None,
                train_Yvar=self._make_train_Yvar(y_r),
            ).to(self.device)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

            loaded = False
            if model_states and len(model_states) > r:
                try:
                    gp.load_state_dict(model_states[r])
                    loaded = True
                except Exception as exc:
                    print(f"[GP.fit] Warning: model[{r}] state failed: {exc}. Refitting.")

            if not loaded:
                fit_gpytorch_mll(mll)
                all_restored = False

            self.models_list.append(gp)

        self.model_indep = ModelListGP(*self.models_list)
        self.model = None
        return all_restored


    def predict(
        self,
        x_new: ArrayOrTensor,
        return_torch: bool = True,
        return_posterior: bool = False,
        observation_noise: bool = False,
    ):
        X_new = self._ensure_tensor(x_new)

        if self.model_type == "independent" and self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.model_type == "mo_indep" and self.model_indep is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_new_norm = X_new if self.no_std_norm else self._x_normalizer.transform(X_new)
        add_noise = observation_noise and self.noise

        if self.model_type == "independent":
            posterior = self.model.posterior(X_new_norm, observation_noise=add_noise)
        elif self.model_type == "mo_indep":
            posterior = self.model_indep.posterior(X_new_norm, observation_noise=add_noise)

        y_mean_std = posterior.mean
        y_var_std = posterior.variance

        mvn_std = posterior.mvn
        mvn = self._unstandardize_mvn(mvn_std)

        if return_posterior:
            return mvn

        if self.no_std_norm:
            y_mean = y_mean_std
            y_var = y_var_std
        else:
            std_v = self._y_standardizer.std_.view(1, -1)
            mean_v = self._y_standardizer.mean_.view(1, -1)
            y_mean = y_mean_std * std_v + mean_v
            y_var = y_var_std * (std_v ** 2)

        out = {
            "y_mean_std":y_mean_std,
            "y_var_std": y_var_std,
            "y_mean": y_mean,
            "y_var": y_var,
            "posterior_mvn_std": mvn_std,
            "posterior_mvn": mvn,
        }

        if not return_torch:
            out = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for k, v in out.items()}

        return out


    def _get_obs_noise_var_std(self) -> Optional[torch.Tensor]:
        """
        Returns the learned noise variances (s for p>1) in STANDARDISED output space,
        shaped (1, R) for mo_indep, or scalar for independent.
        Returns None if noise=False or noise is unavailable.
        """
        if not self.noise:
            return None

        if self.model_type == "independent" and self.model is not None:
            try:
                sigma2 = self.model.likelihood.noise.squeeze() # scalar
                return sigma2.view(1, 1)
            except AttributeError:
                return None

        if self.model_type == "mo_indep" and self.models_list is not None:
            try:
                sigmas = torch.stack([
                    model_r.likelihood.noise.squeeze()
                    for model_r in self.models_list
                ]).view(1, -1)   # (1, R)
                return sigmas
            except AttributeError:
                return None

        return None

    def grad_posterior(
        self,
        X: ArrayOrTensor,
        only_mean: bool = False,
        *,
        chunked: bool = False,
        chunk_size: Optional[int] = None,
        chunk_kmeans: str = "default",
        return_dist: bool = False,
    ) -> List[MultivariateNormal]:

        X = self._ensure_tensor(X)
        if X.dim() != 2:
            raise ValueError("X must be 2D (N, d).")

        N, d = X.shape

        if self.model_type == "independent":
            result = self._grad_posterior_independent(
                X, only_mean=only_mean, chunked=chunked,
                chunk_size=chunk_size, chunk_kmeans=chunk_kmeans,
                return_dist=return_dist,
            )
        elif self.model_type == "mo_indep":
            result = self._grad_posterior_mo_indep(X, only_mean=only_mean)
        elif self.model_type == "icm":
            result = self._grad_posterior_icm(X, only_mean=only_mean)
        else:
            raise NotImplementedError(
                f"grad_posterior not implemented for model_type='{self.model_type}'."
            )

        return result

    def _grad_posterior_independent(
        self,
        X: torch.Tensor,
        only_mean: bool = False,
        chunked: bool = False,
        chunk_size: Optional[int] = None,
        chunk_kmeans: str = "default",
        return_dist: bool = False,
    ):
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.output_dim and self.output_dim != 1:
            raise NotImplementedError("grad_posterior (scalar) supports output_dim == 1 only.")
        if chunked and only_mean:
            raise ValueError("chunked=True requires only_mean=False.")
        if return_dist and not chunked:
            raise ValueError("return_dist=True only supported with chunked=True.")

        N, d = X.shape

        if self.no_std_norm:
            Xt = X
            inv_rng = torch.ones(d, device=self.device, dtype=self.dtype)
            std_y = torch.ones(1, device=self.device, dtype=self.dtype)
        else:
            Xt = self._x_normalizer.transform(X)
            inv_rng = (1.0 / self._x_normalizer._rng.squeeze()).to(self.device, self.dtype)
            std_y = self._y_standardizer.std_.squeeze().to(self.device, self.dtype)
            if std_y.dim() == 0:
                std_y = std_y.unsqueeze(0)

        if inv_rng.dim() == 0:
            inv_rng = inv_rng.unsqueeze(0)

        # scale_per_dim[k] = σ_y / rng_k
        scale_per_dim = (std_y.view(-1)[0] * inv_rng).to(self.device, self.dtype)  # (d,)

        if chunked:
            selection_alg = {
                "default": "kmeans",
                "balanced": "kmeans-equal",
                "kmeans-equal": "kmeans-equal",
                "random": "random",
            }.get(chunk_kmeans.lower() if isinstance(chunk_kmeans, str) else chunk_kmeans)
            if selection_alg is None:
                raise ValueError("chunk_kmeans must be one of {default, balanced, random}.")

            chunk_payload = batched_posterior_derivative_joint_fantasize_chunked_kmeans(
                self.model, Xt,
                chunk_size=chunk_size,
                selection_alg=selection_alg,
                return_indices=return_dist,
            )
            if return_dist:
                chunk_results_std, chunk_indices = chunk_payload
            else:
                chunk_results_std = chunk_payload
                chunk_indices = None

            scaled_chunks = []
            for mean_std, cov_std in chunk_results_std:
                chunk_pts = mean_std.shape[-1] // d
                chunk_scale = scale_per_dim.repeat(chunk_pts).view(1, 1, chunk_pts * d)
                mean_orig = mean_std * chunk_scale
                cov_scale = chunk_scale.unsqueeze(-1) * chunk_scale.unsqueeze(-2)
                cov_orig = cov_std  * cov_scale
                scaled_chunks.append((mean_orig, cov_orig))

            if return_dist:
                if chunk_indices is None:
                    raise RuntimeError("Chunk indices missing for return_dist assembly.")
                mvn = self._assemble_chunked_distribution(
                    scaled_chunks, chunk_indices, total_points=N, dim=d
                )
                return [mvn]
            return scaled_chunks

        if only_mean:
            mean_std = batched_posterior_derivative_joint_fantasize(
                self.model, Xt, return_full=False
            )
            scale_rep = scale_per_dim.repeat(N).view(1, 1, N * d)
            return (mean_std * scale_rep).view(N, d)

        mvn_norm = batched_posterior_derivative_joint_fantasize(self.model, Xt)
        mean_std, cov_std = mvn_norm.mean, mvn_norm.covariance_matrix

        scale_rep = scale_per_dim.repeat(N).to(self.device, self.dtype)
        mean_scale = scale_rep.view(1, 1, N * d)
        mean_orig = mean_std * mean_scale

        cov_scale = mean_scale.unsqueeze(-1) * mean_scale.unsqueeze(-2)
        cov_orig = cov_std * cov_scale

        return [MultivariateNormal(mean_orig, covariance_matrix=cov_orig)]


    def _grad_posterior_mo_indep(
        self,
        X: torch.Tensor,
        only_mean: bool = False,
    ) -> List[MultivariateNormal]:
        """
        Joint Jacobian posterior for R independent GPs.
            Z = [∇f_1(x_1), …, ∇f_1(x_N),  block 0: shape N*d
                ∇f_2(x_1), …, ∇f_2(x_N),  block 1: shape N*d
                           …
                ∇f_R(x_1), …, ∇f_R(x_N)] block R-1: shape N*d
            \in\mathbb{R}^{ R * N * d}
        Covariance is block-diagonal (independence across outputs):
            Σ_J = blockdiag(Σ^∇_1, Σ^∇_2, …, Σ^∇_R)
        mean : (n_fant, B, R*N*d)
        covariance_matrix   : (n_fant, B, R*N*d, R*N*d)[block-diagonal]
        """
        if self.models_list is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        N, d = X.shape
        R = self.output_dim

        if self.no_std_norm:
            Xt = X
            inv_rng = torch.ones(d, device=self.device, dtype=self.dtype)
            std_y = torch.ones(R, device=self.device, dtype=self.dtype)
        else:
            Xt = self._x_normalizer.transform(X)
            inv_rng = (1.0 / self._x_normalizer._rng.squeeze()).to(self.device, self.dtype)
            std_y = self._y_standardizer.std_.view(-1).to(self.device, self.dtype)
            if std_y.shape[0] != R:
                raise RuntimeError(
                    f"Standardizer has {std_y.shape[0]} outputs but output_dim={R}."
                )
        if inv_rng.dim() == 0:
            inv_rng = inv_rng.unsqueeze(0)

        all_means: List[torch.Tensor] = []
        all_covs : List[torch.Tensor] = []

        for r, gp_r in enumerate(self.models_list):
            scale_r = (std_y[r] * inv_rng).to(self.device, self.dtype)
            scale_rep = scale_r.repeat(N)# (N*d,)

            if only_mean:
                mean_std = batched_posterior_derivative_joint_fantasize(
                    gp_r, Xt, return_full=False
                )
                mean_orig = mean_std * scale_rep.view(1, 1, N * d)
                all_means.append(mean_orig)

            else:
                mvn_r = batched_posterior_derivative_joint_fantasize(gp_r, Xt)
                mean_std = mvn_r.mean # (n_fant, B, N*d)
                cov_std = mvn_r.covariance_matrix # (n_fant, B, N*d, N*d)

                mean_orig = mean_std * scale_rep.view(1, 1, N * d)
                cov_scale = (scale_rep.view(1, 1, N * d, 1)
                              * scale_rep.view(1, 1, 1, N * d))
                cov_orig = cov_std * cov_scale

                all_means.append(mean_orig)
                all_covs.append(cov_orig)

        joint_mean = torch.cat(all_means, dim=-1) # (n_fant, B, R*N*d)

        if only_mean:
            n_fant, B_dim, _ = joint_mean.shape
            return joint_mean.view(n_fant, B_dim, R, N, d)

        n_fant, B_dim = all_covs[0].shape[:2]
        total_dim = R * N * d
        joint_cov = torch.zeros(
            n_fant, B_dim, total_dim, total_dim,
            device=self.device, dtype=self.dtype,
        )
        for r in range(R):
            s, e = r * N * d, (r + 1) * N * d
            joint_cov[:, :, s:e, s:e] = all_covs[r]

        joint_cov = 0.5 * (joint_cov + joint_cov.transpose(-1, -2))
        joint_cov = joint_cov + 1e-6 * torch.eye(
            total_dim, device=self.device, dtype=self.dtype
        ).expand(n_fant, B_dim, total_dim, total_dim)

        return [MultivariateNormal(joint_mean, joint_cov)]

    def _assemble_chunked_distribution(
        self,
        scaled_chunks: List[Tuple[torch.Tensor, torch.Tensor]],
        chunk_indices: List[torch.Tensor],
        *,
        total_points: int,
        dim: int,
    ) -> MultivariateNormal:
        if len(scaled_chunks) != len(chunk_indices):
            raise ValueError("Chunk data and index lists must share the same length.")
        if not scaled_chunks:
            raise ValueError("No chunked data available to assemble.")

        sample_mean = scaled_chunks[0][0]
        n_fant, B, _ = sample_mean.shape
        device, dtype = sample_mean.device, sample_mean.dtype
        total_d = total_points * dim

        mean_full = torch.zeros(n_fant, B, total_d,device=device,dtype=dtype)
        cov_full = torch.zeros(n_fant, B, total_d,total_d, device=device, dtype=dtype)
        arange_d = torch.arange(dim, device=device, dtype=torch.long)

        for (mean_chunk, cov_chunk), idx in zip(scaled_chunks, chunk_indices):
            idx = idx.to(device=device, dtype=torch.long)
            if idx.numel() == 0:
                continue
            flat_idx = (idx.unsqueeze(-1) * dim + arange_d).reshape(-1)
            mean_full[..., flat_idx] = mean_chunk
            try:
                grid_i, grid_j = torch.meshgrid(flat_idx, flat_idx, indexing="ij")
            except TypeError:
                grid_i, grid_j = torch.meshgrid(flat_idx, flat_idx)
            cov_full[:, :, grid_i, grid_j] = cov_chunk

        return MultivariateNormal(mean_full, cov_full)
    def fantasize(
        self,
        Xcond: ArrayOrTensor,
        n_fantasies: int = 8,
        sampler: Optional["SobolQMCNormalSampler"] = None,
    ) -> "GP":
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_fantasies]))

        Xcond=self._ensure_tensor(Xcond)
        Xcond_norm=Xcond if self.no_std_norm else self._x_normalizer.transform(Xcond)

        fant_gp = GP(
            model_type=self.model_type,
            device=str(self.device),
            dtype=self.dtype,
            no_std_norm=self.no_std_norm,
            noise=self.noise,
        )
        fant_gp._x_normalizer = self._x_normalizer
        fant_gp._y_standardizer = self._y_standardizer
        fant_gp.X_train = self.X_train
        fant_gp.Y_train = self.Y_train
        fant_gp.input_dim = self.input_dim
        fant_gp.output_dim = self.output_dim

        if self.model_type == "independent":
            if self.model is None:
                raise RuntimeError("Model not trained. Call fit() first.")
            fant_gp.model = self.model.fantasize(X=Xcond_norm, sampler=sampler)
            fant_gp.models_list = None
            fant_gp.model_indep = None

        elif self.model_type == "mo_indep":
            if self.models_list is None:
                raise RuntimeError("Model not trained. Call fit() first.")
            fant_models = [
                gp_r.fantasize(X=Xcond_norm, sampler=sampler)
                for gp_r in self.models_list
            ]
            fant_gp.models_list = fant_models
            fant_gp.model_indep = None
            fant_gp.model = None

        return fant_gp


    def state_dict(self) -> Dict[str, Any]:
        state = {
            "model_type":self.model_type,
            "no_std_norm":self.no_std_norm,
            "noise":self.noise,          
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
        }

        if self.model_type == "independent" and self.model is not None:
            state["model_state"] = self.model.state_dict()
        elif self.model_type == "mo_indep" and self.models_list is not None:
            state["model_states"] = [gp.state_dict() for gp in self.models_list]

        if not self.no_std_norm:
            state.update({
                "x_min":(self._x_normalizer.min_.detach().cpu().numpy()
                           if self._x_normalizer and self._x_normalizer.min_ is not None else None),
                "x_max":(self._x_normalizer.max_.detach().cpu().numpy()
                           if self._x_normalizer and self._x_normalizer.max_ is not None else None),
                "y_mean":(self._y_standardizer.mean_.detach().cpu().numpy()
                           if self._y_standardizer and self._y_standardizer.mean_ is not None else None),
                "y_std":(self._y_standardizer.std_.detach().cpu().numpy()
                           if self._y_standardizer and self._y_standardizer.std_ is not None else None),
            })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> "GP":

        if state is None:
            return self

        saved_model_type = state.get("model_type")
        if saved_model_type is not None and saved_model_type != self.model_type:
            self.model_type = saved_model_type

        self.input_dim = state.get("input_dim", self.input_dim)
        self.output_dim=state.get("output_dim", self.output_dim)
        self.no_std_norm=state.get("no_std_norm", self.no_std_norm)
        self.noise= state.get("noise", self.noise)

        if not self.no_std_norm:
            if self._x_normalizer is None:
                self._x_normalizer = MinMaxNormalizer(device=self.device, dtype=self.dtype)
            if self._y_standardizer is None:
                self._y_standardizer = Standardizer(device=self.device, dtype=self.dtype)

            if state.get("x_min") is not None and state.get("x_max") is not None:
                self._x_normalizer.min_ = torch.from_numpy(state["x_min"]).to(self.device, self.dtype)
                self._x_normalizer.max_ = torch.from_numpy(state["x_max"]).to(self.device, self.dtype)
                self._x_normalizer._rng = self._x_normalizer.max_ - self._x_normalizer.min_

            if state.get("y_mean") is not None and state.get("y_std") is not None:
                self._y_standardizer.mean_ = torch.from_numpy(state["y_mean"]).to(self.device, self.dtype)
                self._y_standardizer.std_ = torch.from_numpy(state["y_std"]).to(self.device, self.dtype)

        return self
    
    def frobenius_jacobian_variance(self, X: ArrayOrTensor) -> torch.Tensor:
        """
        Var(||Jac_η(X_s)||²_F)  under the GP posterior.
        total : (n_fant, B) 
        """
        X = self._ensure_tensor(X)
        N, d = X.shape
        Nd = N * d

        if self.no_std_norm:
            Xt = X
            inv_rng = torch.ones(d, device=self.device, dtype=self.dtype)
            std_y = torch.ones(self.output_dim, device=self.device, dtype=self.dtype)
        else:
            Xt = self._x_normalizer.transform(X)
            inv_rng = (1.0 / self._x_normalizer._rng.squeeze()).to(self.device, self.dtype)
            std_y = self._y_standardizer.std_.view(-1).to(self.device, self.dtype)

        total = None

        for r, gp_r in enumerate(self.models_list):

            scale = (std_y[r] * inv_rng).repeat(N)# (N·d,)

            mvn_r = batched_posterior_derivative_joint_fantasize(gp_r, Xt) 
            mu_std=mvn_r.mean
            Sig_std=mvn_r.covariance_matrix

            S_outer = scale.unsqueeze(-1) * scale.unsqueeze(-2)  # (Nd, Nd)
            Sig_r = Sig_std * S_outer# (n_fant, B, Nd, Nd)

            mu_r = mu_std * scale # (n_fant, B, Nd)

            tr_Sig2 = (Sig_r ** 2).sum(dim=(-2, -1))# (n_fant, B)

            Sig_mu = torch.einsum('...ij,...j->...i', Sig_r, mu_r) # (n_fant, B, Nd)
            mu_Sig_mu = torch.einsum('...i,...i->...', mu_r, Sig_mu) # (n_fant, B)

            block = 2.0 * tr_Sig2 + 4.0 * mu_Sig_mu

            total = block if total is None else total + block

        return total # (n_fant, B)

    def _get_single_output_models(self):
        if self.model_type == "independent":
            if self.model is None:
                raise RuntimeError("Model not trained. Call fit() first.")
            return [self.model]
        if self.model_type == "mo_indep":
            if self.models_list is None:
                raise RuntimeError("Model not trained. Call fit() first.")
            return self.models_list
        raise NotImplementedError(
            "Closed-form Jacobian ACQF is implemented for independent/mo_indep only."
        )

    def frobenius_jacobian_variance_generic(self, X: ArrayOrTensor) -> torch.Tensor:
        """
        Generic Var(||Jac(X)||_F^2) using grad_posterior.
        Works for current and fantasized models, scalar and mo_indep.

        Output:
          - current model: scalar or (B,)
          - fantasy model: (n_fant, B)
        """
        X = self._ensure_tensor(X)
        mvn_list = self.grad_posterior(X, only_mean=False)
        if len(mvn_list) != 1:
            raise RuntimeError("Expected a single joint MVN from grad_posterior.")
        mvn = mvn_list[0]
        return posterior_quadratic_form_variance(mvn.mean, mvn.covariance_matrix)

    def _jacobian_mean_blocks_orig(self, Xs: ArrayOrTensor):
        Xs = self._ensure_tensor(Xs)
        Jmean = self.grad_posterior(Xs, only_mean=True)
        if self.model_type == "independent":
            return [Jmean.reshape(-1)]
        Jmean = Jmean.squeeze(0).squeeze(0)  # (R,N,d)
        return [Jmean[r].reshape(-1) for r in range(self.output_dim)]

    def _training_cov_with_noise_std_single_output(self, gp_r):
        Xtr = gp_r.train_inputs[0]
        if Xtr.ndim != 2:
            raise ValueError("Expected train_inputs[0] shape (n,d).")
        n = Xtr.shape[0]
        K = gp_r.covar_module(Xtr, Xtr).to_dense()
        eye = torch.eye(n, dtype=K.dtype, device=K.device)
        if self.noise:
            sigma2 = gp_r.likelihood.noise.squeeze()
            K = K + sigma2 * eye
        else:
            K = K + 1e-5 * eye
        return K

    def _predictive_obs_variance_std_single_output(
        self,
        gp_r,
        Xcand_norm: torch.Tensor,
        include_obs_noise: bool = True,
    ):
        post = gp_r.posterior(Xcand_norm, observation_noise=include_obs_noise)
        return post.variance.squeeze(-1).squeeze(-1)

    def _crosscov_grad_obs_std_single_output(
        self,
        gp_r,
        Xs: ArrayOrTensor,
        Xcand: ArrayOrTensor,
    ) -> torch.Tensor:
        """
        c_std(x) = Cov(vec(d f_tilde / d x_tilde at Xs), y_tilde(x) | D)
        in standardized-output / normalized-input coordinates.

        Returns:
            c_std : (B, N*d)
        """
        Xs = self._ensure_tensor(Xs)
        Xcand = self._ensure_tensor(Xcand)

        if Xcand.dim() == 3:
            Xcand = Xcand.squeeze(1)
        if Xcand.dim() != 2:
            raise ValueError("Xcand must be (B,d) or (B,1,d).")

        N, d = Xs.shape

        if self.no_std_norm:
            Xs_n = Xs
            Xc_n = Xcand
        else:
            Xs_n = self._x_normalizer.transform(Xs)
            Xc_n = self._x_normalizer.transform(Xcand)

        Xtr = gp_r.train_inputs[0]
        if Xtr.ndim != 2:
            raise ValueError("Closed-form helper expects current model train_inputs[0] shape (n,d).")
        n = Xtr.shape[0]

        K = self._training_cov_with_noise_std_single_output(gp_r)
        eye = torch.eye(n, dtype=K.dtype, device=K.device)
        L = torch.linalg.cholesky(K + 1e-6 * eye)

        def solveK(rhs):
            return torch.cholesky_solve(rhs, L)

        lengthscales = gp_r.covar_module.base_kernel.lengthscale.squeeze().to(Xs_n)
        outputscale = gp_r.covar_module.outputscale.squeeze().to(Xs_n)

        grad_train = matern52_grad_first_arg(Xs_n, Xtr, lengthscales, outputscale) # (N,n,d)
        G_std = grad_train.permute(0, 2, 1).reshape(N * d, n)  # (N*d,n)

        grad_x = matern52_grad_first_arg(Xs_n, Xc_n, lengthscales, outputscale) # (N,B,d)
        grad_x_flat = grad_x.permute(1, 0, 2).reshape(Xc_n.shape[0], N * d)# (B,N*d)

        k_X_x = gp_r.covar_module(Xtr, Xc_n).to_dense() # (n,B)
        Kinv_k = solveK(k_X_x)# (n,B)
        corr = (G_std @ Kinv_k).transpose(0, 1)# (B,N*d)

        return grad_x_flat - corr

    def _rescale_crosscov_and_var_to_original(
        self,
        c_std: torch.Tensor,
        v_std: torch.Tensor,
        output_index: int,
        N: int,
    ):
        if self.no_std_norm:
            return c_std, v_std
        inv_rng = (1.0 / self._x_normalizer._rng.squeeze()).to(c_std.device, c_std.dtype)
        std_y = self._y_standardizer.std_.view(-1)[output_index].to(c_std.device, c_std.dtype)
        c_scale = (std_y ** 2) * inv_rng
        c_scale = c_scale.repeat(N)

        c_orig = c_std * c_scale.unsqueeze(0)
        v_orig = v_std * (std_y ** 2)
        return c_orig, v_orig

    def closed_form_jac_acqf_single(
        self,
        global_points: ArrayOrTensor,
        Xcand: ArrayOrTensor,
        include_obs_noise: bool = True,
    ) -> torch.Tensor:
        """
        !!! Only for q=1 !!!! 

        alpha(x) = sum_r [ 2 \Vert c_r \Vert ^4 / v_r^2 + 4 (mu_r^T c_r)^2 / v_r ]

        Inputs:
            global_points Xs: (N,d)
            Xcand: (B,d) or (B,1,d)
            include_obs_noise: Bool

        Returns:
            acq : (B,)
        """
        global_points = self._ensure_tensor(global_points)
        Xcand = self._ensure_tensor(Xcand)
        if Xcand.dim() == 3:
            Xcand = Xcand.squeeze(1)

        gps = self._get_single_output_models()
        mu_blocks = self._jacobian_mean_blocks_orig(global_points)

        if self.no_std_norm:
            Xcand_n = Xcand
        else:
            Xcand_n = self._x_normalizer.transform(Xcand)

        N = global_points.shape[0]
        B = Xcand.shape[0]
        total = torch.zeros(B, dtype=self.dtype, device=self.device)

        for r, gp_r in enumerate(gps):
            c_std = self._crosscov_grad_obs_std_single_output(gp_r, global_points, Xcand)
            v_std = self._predictive_obs_variance_std_single_output(
                gp_r,
                Xcand_n,
                include_obs_noise=include_obs_noise,
            )
            c, v = self._rescale_crosscov_and_var_to_original(
                c_std, v_std, output_index=r, N=N
            )

            mu = mu_blocks[r]#(N*d,)
            c_norm_sq = (c ** 2).sum(dim=-1) #(B,)
            mu_dot_c = c @ mu # (B,)

            total = total + 2.0 * (c_norm_sq ** 2) / (v ** 2) + 4.0 * (mu_dot_c ** 2) / v
        return total

    def closed_form_jac_acqf_single2(
        self,
        global_points: ArrayOrTensor,
        Xcand: ArrayOrTensor,
        include_obs_noise: bool = True,
        lam: Optional[torch.Tensor] = None,  # (K,) PCA eigenvalues (for the Lambda version of JAcques !); None = classico
    ) -> torch.Tensor:
        """
        alpha(x) = sum_r w_r [ 2||c_r||^4/v_r^2 + 4(mu_r'c_r)^2/v_r ]

        w_r = lam[r]**2 if lam is provided  (lambda^2-weighted)
        w_r = 1 if lam is None (original Jacques)
        """
        global_points = self._ensure_tensor(global_points)
        Xcand = self._ensure_tensor(Xcand)
        if Xcand.dim() == 3:
            Xcand = Xcand.squeeze(1)

        gps = self._get_single_output_models()
        mu_blocks = self._jacobian_mean_blocks_orig(global_points)

        if self.no_std_norm:
            Xcand_n = Xcand
        else:
            Xcand_n = self._x_normalizer.transform(Xcand)

        N = global_points.shape[0]
        B = Xcand.shape[0]
        total = torch.zeros(B, dtype=self.dtype, device=self.device)

        if lam is not None:
            lam = lam.to(dtype=self.dtype, device=self.device)
            weights = lam ** 2                            # (K,)
        else:
            weights = torch.ones(len(gps), dtype=self.dtype, device=self.device)

        for r, gp_r in enumerate(gps):
            c_std = self._crosscov_grad_obs_std_single_output(gp_r, global_points, Xcand)
            v_std = self._predictive_obs_variance_std_single_output(
                gp_r, Xcand_n, include_obs_noise=include_obs_noise,
            )
            c, v = self._rescale_crosscov_and_var_to_original(
                c_std, v_std, output_index=r, N=N
            )
            mu = mu_blocks[r] # (N*d,)
            c_norm_sq= (c ** 2).sum(dim=-1) # (B,)
            mu_dot_c = c @ mu # (B,)

            total += weights[r] * (2.0 * c_norm_sq**2 / v**2 + 4.0 * mu_dot_c**2 / v) # (B,)

        return total

    def closed_form_jac_acqf_poincare_single(
        self,
        global_points,
        Xcand,
        weight_vector,
        include_obs_noise: bool = True,
    ):
        if self.model_type != "independent":
            raise NotImplementedError("This version is for scalar output only.")

        global_points = self._ensure_tensor(global_points)
        Xcand = self._ensure_tensor(Xcand)
        weight_vector = self._ensure_tensor(weight_vector)

        if Xcand.dim() == 3:
            Xcand = Xcand.squeeze(1)

        mu = self._jacobian_mean_blocks_orig(global_points)[0]   # (N*d,)
        gp_r = self.model
        N = global_points.shape[0]

        if self.no_std_norm:
            Xcand_n = Xcand
        else:
            Xcand_n = self._x_normalizer.transform(Xcand)

        c_std = self._crosscov_grad_obs_std_single_output(gp_r, global_points, Xcand)
        v_std = self._predictive_obs_variance_std_single_output(
            gp_r, Xcand_n, include_obs_noise=include_obs_noise
        )
        c, v = self._rescale_crosscov_and_var_to_original(c_std, v_std, output_index=0, N=N)

        Wc = c * weight_vector.unsqueeze(0)   # (B, N*d)
        Wmu = mu * weight_vector# (N*d,)

        cWc = (c * Wc).sum(dim=-1)# c^T W c
        muWc = (Wc @ mu)# mu^T W c

        return 2.0 * (cWc ** 2) / (v ** 2) + 4.0 * (muWc ** 2) / v