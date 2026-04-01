from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import List, Optional, Tuple,Sequence, Dict, Any

from torch import Tensor
import torch
import math
DistSpec = Dict[str, Any]

class Gsobol(SyntheticTestFunction):
    r"""Gsobol test function.

    d-dimensional function (usually evaluated on `[0, 1]^d`):

        f(x) = Prod_{i=1}\^{d} ((\|4x_i-2\|+a_i)/(1+a_i)), a_i >=0

    common combinations of dimension and a vector:

        dim=8, a= [0, 1, 4.5, 9, 99, 99, 99, 99]
        dim=6, a=[0, 0.5, 3, 9, 99, 99]
        dim = 15, a= [1, 2, 5, 10, 20, 50, 100, 500, 1000, ..., 1000]

    Proposed to test sensitivity analysis methods
    First order Sobol indices have closed form expression S_i=V_i/V with :

        V_i= 1/(3(1+a_i)\^2)
        V= Prod_{i=1}\^{d} (1+V_i) - 1
    From Belekaria et al. 2024
    
    """

    def __init__(
        self,
        dim: int,
        a: List = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            dim: Dimensionality of the problem. If 6, 8, or 15, will use standard a.
            a: a parameter, unless dim is 6, 8, or 15.
            noise_std: Standard deviation of observation noise.
            negate: Return negatie of function.

        From Belakaria et al. 2024

        """
        self._optimizers = None
        self.dim = dim
        self.continuous_inds = list(range(self.dim))
        self.dists = [{"type": "unif", "a": 0., "b": 1.}]*self.dim
        self._bounds = [(0, 1) for _ in range(self.dim)]
        if dim == 6:
            self.a = [0, 0.5, 3, 9, 99, 99]
            self.dgsm_gradient_square = [
                1.8814e01,
                9.7107e00,
                1.5355e00,
                2.4996e-01,
                2.5077e-03,
                2.5077e-03,
            ]
        elif dim==2:
            self.a = [0.1, 1.9]
        elif dim == 8:
            self.a = [0, 1, 4.5, 9, 99, 99, 99, 99]
            self.dgsm_gradient_square = [
                1.7578e01,
                5.4097e00,
                7.6645e-01,
                2.3358e-01,
                2.3436e-03,
                2.3436e-03,
                2.3436e-03,
                2.3436e-03,
            ]
        elif dim == 10:
            self.a = [0, 0, 6.52, 6.52, 6.52, 6.52, 6.52, 6.52, 6.52, 6.52]
            self.dgsm_gradient_square = [
                22.3485,
                22.3561,
                0.5241,
                0.5240,
                0.5241,
                0.5240,
                0.5241,
                0.5241,
                0.5241,
                0.5240,
            ]
        elif dim == 15:
            self.a = [
                1,
                2,
                5,
                10,
                20,
                50,
                100,
                500,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
            ]
            self.dgsm_gradient_square = [
                4.2009e00,
                1.9506e00,
                5.0098e-01,
                1.5003e-01,
                4.1247e-02,
                6.9977e-03,
                1.7844e-03,
                7.2523e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
            ]
        else:
            self.a = a
        self.optimal_sobol_indicies()
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def optimal_sobol_indicies(self):
        vi = []
        dim = self.dim
        for i in range(dim):
            vi.append(1 / (3 * ((1 + self.a[i]) ** 2)))
        self.vi = Tensor(vi)
        self.V = torch.prod((1 + self.vi)) - 1
        self.si = self.vi / self.V
        si_t = []
        for i in range(dim):
            si_t.append(
                (
                    self.vi[i]
                    * torch.prod(self.vi[:i] + 1)
                    * torch.prod(self.vi[i + 1 :] + 1)
                )
                / self.V
            )
        self.si_t = Tensor(si_t)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        t = 1
        dim = self.dim
        for i in range(dim):
            t = t * (torch.abs(4 * X[..., i] - 2) + self.a[i]) / (1 + self.a[i])
        return t        


class Lim(SyntheticTestFunction):

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            dim: Dimensionality of the problem. If 6, 8, or 15, will use standard a.
            a: a parameter, unless dim is 6, 8, or 15.
            noise_std: Standard deviation of observation noise.
            negate: Return negatie of function.
        """
        self._optimizers = None
        self.dim = 2
        self.continuous_inds = list(range(self.dim))
        self.dists = [{"type": "unif", "a": 0., "b": 1.}]*self.dim
        self._bounds = [(0, 1) for _ in range(self.dim)]
        
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def _evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        x1, x2 = X[..., 0], X[..., 1]
        return ((30 + 5*x1*torch.sin(5*x1)) * (4 + torch.exp(-5*x2)) - 100) / 6
    

class IshigamiBranin(SyntheticTestFunction):
    """
    f : R^3 _> R^2
      f_1(x) = sin(x1) + 7·sin^2(x2) + 0.1·x3^4·sin(x1)
      f_2(x) = Branin(x1, x2)
    """
    dim: int = 3
    num_outputs: int = 2
    _bounds: List[Tuple[float, float]] = [
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
    ]
    _a = 1.0
    _b = 5.1 / (4 * math.pi ** 2)
    _c = 5.0 / math.pi
    _r = 6.0
    _s = 10.0
    _t = 1.0 / (8 * math.pi)
    _output_stds = [3.72, 53.9] # For better sense between both different outut scal e! 
    def __init__(
        self,
        noise_std: Optional[float] = None,
        noise_std_rel: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.continuous_inds = list(range(3))
        self.dists: Sequence[DistSpec] = [
            {"type": "uniform", "a": -math.pi, "b": math.pi},
            {"type": "uniform", "a": -math.pi, "b": math.pi},
            {"type": "uniform", "a": -math.pi, "b": math.pi},
        ]
        self._bounds = [(-math.pi, math.pi)] * self.dim
        self._noise_std_rel = noise_std_rel
        super().__init__(noise_std=noise_std, negate=negate)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2, x3 = X[..., 0], X[..., 1], X[..., 2]

        f1 = (
            torch.sin(x1)
            + 7.0 * torch.sin(x2).pow(2)
            + 0.1 * x3.pow(4) * torch.sin(x1)
        )

        u = (x1 + math.pi) / (2 * math.pi) * 15.0 - 5.0 # [-5, 10]
        v = (x2 + math.pi) / (2 * math.pi) * 15.0  # [0,  15]
        f2 = (
            self._a * (v - self._b * u.pow(2) + self._c * u - self._r).pow(2)
            + self._s * (1.0 - self._t) * torch.cos(u)
            + self._s
        )

        return torch.stack([f1, f2], dim=-1)

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        Y = self.evaluate_true(X)

        if noise and self._noise_std_rel is not None:
            stds = torch.tensor(
                [s * self._noise_std_rel for s in self._output_stds],
                dtype=X.dtype, device=X.device
            )
            Y = Y + torch.randn_like(Y) * stds # per-output noise

        elif noise and self.noise_std is not None:
            Y = Y + torch.randn_like(Y) * self.noise_std

        return Y
    
    def jacobian(self, X: Tensor) -> Tensor:
        x1, x2, x3 = X[..., 0], X[..., 1], X[..., 2]
        J = torch.zeros(*X.shape[:-1], 2, 3, dtype=X.dtype, device=X.device)

        J[..., 0, 0] = torch.cos(x1) * (1.0 + 0.1 * x3.pow(4))
        J[..., 0, 1] = 7.0 * torch.sin(2.0 * x2)
        J[..., 0, 2] = 0.4 * x3.pow(3) * torch.sin(x1)

        du_dx1 = 15.0 / (2.0 * math.pi)
        dv_dx2 = 15.0 / (2.0 * math.pi)
        u = (x1 + math.pi) / (2 * math.pi) * 15.0 - 5.0
        v = (x2 + math.pi) / (2 * math.pi) * 15.0
        truc = v - self._b * u.pow(2) + self._c * u - self._r

        J[..., 1, 0] = (
            2.0 * self._a * truc * (-2.0 * self._b * u + self._c)
            - self._s * (1.0 - self._t) * torch.sin(u)
        ) * du_dx1
        J[..., 1, 1] = 2.0 * self._a * truc * dv_dx2

        return J