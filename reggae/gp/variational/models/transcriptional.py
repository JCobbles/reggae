from abc import abstractmethod

import torch
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

from torchdiffeq import odeint
from .model import VariationalLFM
from reggae.utilities import softplus, LFMDataset


class TranscriptionalRegulationLFM(VariationalLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, extra_points=2, **kwargs):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, extra_points=extra_points, **kwargs)
        self.initial_condition = Parameter(0.4+torch.rand((self.num_outputs, 1), dtype=torch.float64))
        self.decay_rate = Parameter(1 + torch.rand((self.num_outputs, 1), dtype=torch.float64))
        self.basal_rate = Parameter(torch.rand((self.num_outputs, 1), dtype=torch.float64))
        self.sensitivity = Parameter(1.5 + torch.rand((self.num_outputs, 1), dtype=torch.float64))

    def forward(self, t, h, rtol=1e-4, atol=1e-6, compute_var=False):
        """
        t : torch.Tensor
            Shape (num_times)
        h : torch.Tensor the initial state of the ODE
            Shape (num_genes, 1)
        Returns
        -------
        Returns evolved h across times t.
        Shape (num_genes, num_points).
        """
        self.nfe = 0

        # Precompute variables
        self.Kmm = self.rbf(self.inducing_inputs)
        self.L = torch.cholesky(self.Kmm)
        # self.inv_Kmm = cholesky_inverse(self.L)
        q_cholS = torch.tril(self.q_cholS)
        self.S = torch.matmul(q_cholS, torch.transpose(q_cholS, 1, 2))

        h0 = self.initial_state(h)
        # Integrate forward from the initial positions h.
        h_samples = odeint(self.odefunc, h0, t, method='dopri5', rtol=rtol, atol=atol)  # (T, S, num_outputs, 1)

        h_avg = torch.mean(h_samples, dim=1)
        h_std = torch.std(h_samples, dim=1)

        h_out = torch.transpose(h_avg, 0, 1)
        h_std = torch.transpose(h_std, 0, 1)
        bd = self.basal_rate / self.decay_rate
        edt = torch.exp(-self.decay_rate * t)
        h_out = bd + (self.initial_condition - bd)*edt + edt * h_out.squeeze(-1)
        h_out = h_out.unsqueeze(-1)

        if compute_var:
            return self.decode(h_out), h_std
        return self.decode(h_out)

    def odefunc(self, t, h):
        """h is of shape (num_samples, num_outputs, 1)"""
        self.nfe += 1
        # if (self.nfe % 100) == 0:
        #     print(t)

        # decay = torch.multiply(self.decay_rate.squeeze(), h.squeeze(-1)).view(self.num_samples, -1, 1)

        q_f = self.get_latents(t.reshape(-1))

        # Reparameterisation trick
        f = q_f.rsample([self.num_samples])  # (S, I, t)

        f = self.G(f)
        if self.extra_points > 0:
            f = f[:, :, self.extra_points]  # get the midpoint
            f = torch.unsqueeze(f, 2)

        # return self.basal_rate + self.sensitivity * f - decay
        return torch.exp(self.decay_rate * t) * f

    @abstractmethod
    def G(self, f):
        """
        Parameters:
            f: (I, T)
        """
        pass


class SingleLinearLFM(TranscriptionalRegulationLFM):

    def G(self, f):
        # I = 1 so just repeat for num_outputs
        return f.repeat(1, self.num_outputs, 1)


class NonLinearLFM(TranscriptionalRegulationLFM):

    def G(self, f):
        # I = 1 so just repeat for num_outputs
        return softplus(f).repeat(1, self.num_outputs, 1)


class ExponentialLFM(TranscriptionalRegulationLFM):

    def G(self, f):
        # I = 1 so just repeat for num_outputs
        return torch.exp(f).repeat(1, self.num_outputs, 1)


class MultiLFM(TranscriptionalRegulationLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, t_observed, fixed_variance=None):
        super().__init__(num_outputs, num_latents, t_inducing, t_observed, fixed_variance=fixed_variance)
        self.w = Parameter(torch.ones((self.num_outputs, self.num_latents), dtype=torch.float64))
        self.w_0 = Parameter(torch.ones((self.num_outputs, 1), dtype=torch.float64))

    def G(self, f):
        p_pos = softplus(f)  # (S, I, extras)
        interactions = torch.matmul(self.w, torch.log(p_pos+1e-50)) + self.w_0  # (J,I)(I,e)+(J,1)
        return torch.sigmoid(interactions)  # TF Activation Function (sigmoid)


class PoissonLFM(TranscriptionalRegulationLFM):
    def __init__(self, num_outputs, num_latents, t_inducing, dataset: LFMDataset, fixed_variance=None):
        super().__init__(num_outputs, num_latents, t_inducing, dataset, fixed_variance=fixed_variance, extra_points=0)

    """Adds poison to the latent forces"""
    def G(self, λ):
        # λ (I, points) is the parameter of the poison distribution
        print('lam shape', λ.shape)
        # f = Poisson(λ).rsample() #  not implemented - no reparam trick for Poisson implemented
        # so we use an approximation as a N(λ, λ)
        f = Normal(λ, λ).rsample()
        return f.repeat(self.num_outputs, 1)
