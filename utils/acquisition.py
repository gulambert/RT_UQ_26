from botorch.acquisition import AcquisitionFunction
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from botorch import settings

from gpytorch.distributions import MultivariateNormal

from torch import Tensor
import torch

def _global_variance(posterior) -> torch.Tensor:
    r"""
    Computes the global variance of the quad foirm: 
         Var = 2 * tr(Sigma_nabla^2) + 4 * mu_nabla^T Sigma_nabla mu_nabla,
    """
    S = posterior.covariance_matrix[0]
    trace_S2 = torch.sum(S * S, dim=(-2, -1))
    m = posterior.mean 
    mSm = torch.einsum('fbi,bij,fbj->fb', m, S, m) 
    mSm_mean = mSm.mean(dim=0) 
    global_var = 2.0 * trace_S2 + 4.0 * mSm_mean #shape (B,)
    return global_var

def _expand_poincare_weights(constants: Tensor, num_points: int) -> Tensor:
    if constants.ndim != 1:
        raise ValueError("Poincaré constants tensor must be one-dimensional.")
    return constants.repeat(num_points)


def _to_local_posterior(posterior, N, d, n_fant = 1):
    mean_full = posterior.mean               # (n_fant, 1, N*d)
    cov_full  = posterior.covariance_matrix  # (n_fant, 1, N*d, N*d)
    mean_full = mean_full.squeeze(1)    # (n_fant, N*d)
    cov_full  = cov_full.squeeze(1)     # (n_fant, N*d, N*d)
    mean_local = mean_full.view(n_fant, N, d)  # (n_fant, N, d)
    cov_local = cov_full.view(n_fant,N, d,N, d)  # (n_fant, N, d, N, d)
    cov_local = cov_local.diagonal(dim1=1, dim2=3)# shape: (n_fant, d, d, N)
    cov_local = cov_local.permute(0, 3, 1, 2)# (n_fant, N, d, d)
    local_post = MultivariateNormal(
    mean_local,
     covariance_matrix=cov_local)
    return local_post

class LocalMaxVar(AcquisitionFunction):
    """Maximise the variance of ||∇η(x)||^2 at the candidate point."""

    def __init__(self, model):
        super().__init__(model=model)
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        Xs = X.squeeze(1)
        N,d = Xs.shape
        _posterior = self.model.grad_posterior(Xs)[0]
        posterior = _to_local_posterior(posterior = _posterior, N = N , d=d)
        return _global_variance(posterior)


class LocalGradVar(AcquisitionFunction):
    """Local variance reduction analogue of GlobalVarianceReduction."""

    def __init__(self, model, num_fantasies: int = 16):
        super().__init__(model=model)
        self.num_fantasies = num_fantasies
        self.sampler = SobolQMCNormalSampler(torch.Size([num_fantasies]))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        Xs = X.squeeze(1)
        N,d = Xs.shape
        _posterior_curr = self.model.grad_posterior(Xs)[0]
        posterior_curr = _to_local_posterior(posterior = _posterior_curr, N = N , d=d, n_fant=1)
        curr_var = _global_variance(posterior_curr)
        with settings.propagate_grads(True):
            fantasy_model = self.model.fantasize(Xcond=Xs, sampler=self.sampler)
            _posterior_fant = fantasy_model.grad_posterior(Xs)[0]
            posterior_fant = _to_local_posterior(posterior = _posterior_fant, N = N , d=d, n_fant=self.num_fantasies)
        lookahead_var = _global_variance(posterior_fant)
        return curr_var - lookahead_var


class GlobalGradVarRed(AcquisitionFunction):
    def __init__(
        self,
        model,
        global_points: Tensor,  # (N, d)
        num_fantasies: int = 16,
    ):
        super().__init__(model=model)
        self.global_points = global_points
        self.num_fantasies = num_fantasies
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_fantasies]))
        with torch.no_grad():
            posterior_joint = self.model.grad_posterior(self.global_points)[0]
            self.curr_var = _global_variance(posterior_joint)
            
    def forward(self, X:Tensor)-> Tensor:
        """
        X : (b, 1, d)
        """
        with settings.propagate_grads(True):
            fantasy_model = self.model.fantasize(Xcond=X, sampler=self.sampler) # X (b, q, d) allows independent conditionning for the fantasy model batch shape (n_fant, b)
            posterior_fant = fantasy_model.grad_posterior(self.global_points)[0]
        lookahead_var = _global_variance(posterior_fant) 
        acqf_val = self.curr_var - lookahead_var #(b,)
        return acqf_val

class Jacques(AcquisitionFunction):
    def __init__(self, model, global_points,include_obs_noise = True):
        super().__init__(model=model)
        self.include_obs_noise = include_obs_noise
        self.register_buffer("global_points", global_points)

    def forward(self, X):
        return self.model.closed_form_jac_acqf_single(
            global_points = self.global_points, 
            Xcand = X, 
            include_obs_noise = self.include_obs_noise
        )

class JacquesLambda(AcquisitionFunction):
    def __init__(self, model, global_points, lam, include_obs_noise=True):
        super().__init__(model=model)
        self.include_obs_noise = include_obs_noise
        self.register_buffer("global_points", global_points)
        self.register_buffer("lam", lam.to(global_points))

    def forward(self, X):
        return self.model.closed_form_jac_acqf_single2(
            self.global_points, X, lam=self.lam,include_obs_noise=self.include_obs_noise
        )
class JacquesPoincare(AcquisitionFunction):
    def __init__(self, model, global_points, poincare_consts, include_obs_noise=True):
        super().__init__(model=model)
        self.include_obs_noise = include_obs_noise
        self.register_buffer("global_points", global_points)
        self.register_buffer(
            "weight_vector",
            _expand_poincare_weights(poincare_consts, global_points.shape[0]),
        )

    def forward(self, X):
        return self.model.closed_form_jac_acqf_poincare_single(
            global_points=self.global_points,
            Xcand=X,
            weight_vector=self.weight_vector,
            include_obs_noise=self.include_obs_noise,
        )