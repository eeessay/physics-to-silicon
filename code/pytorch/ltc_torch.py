# ============================================================
# Author: Member 3 (SHEN)
# EE5311 CA1-21 — PyTorch Implementation of LTC (Aligned to Julia)
#
# This module implements the canonical Liquid Time-Constant (LTC) model:
#
#   dx/dt = -x / τ(x,I) + f(x,I)
#   τ(x,I) = τ₀ + softplus(Wτ·x + Uτ·I + bτ)
#   f(x,I) = tanh(Wf·x + Uf·I + bf)
#
# The implementation follows the formulation used in the
# reference Julia version. Model parameters can be loaded
# from Julia-exported .npy files for cross-framework alignment.
#
# Alignment with the Julia implementation is verified separately.
# ============================================================
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint


class LTCCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, tau0: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        scale = 0.1

        self.W_tau = nn.Parameter(scale * torch.randn(hidden_dim, hidden_dim, dtype=torch.float32))
        self.U_tau = nn.Parameter(scale * torch.randn(hidden_dim, input_dim, dtype=torch.float32))
        self.b_tau = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
        self.tau0 = nn.Parameter(torch.full((hidden_dim,), tau0, dtype=torch.float32))

        self.W_f = nn.Parameter(scale * torch.randn(hidden_dim, hidden_dim, dtype=torch.float32))
        self.U_f = nn.Parameter(scale * torch.randn(hidden_dim, input_dim, dtype=torch.float32))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))

        self.softplus = nn.Softplus()

    def tau(self, x: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        return self.tau0 + self.softplus(self.W_tau @ x + self.U_tau @ I + self.b_tau)

    def f_drift(self, x: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.W_f @ x + self.U_f @ I + self.b_f)

    def ltc_dynamics(self, x: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        tau_val = self.tau(x, I)
        f_val = self.f_drift(x, I)
        return -x / tau_val + f_val


class LTCSequenceODE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = LTCCell(input_dim, hidden_dim)

    def rollout(
        self,
        x0: torch.Tensor,
        I_seq: torch.Tensor,
        t_span: tuple[float, float],
        t_save: torch.Tensor,
    ) -> torch.Tensor:
        device = x0.device
        dtype = x0.dtype

        I_seq = I_seq.to(device=device, dtype=dtype)
        t_save = t_save.to(device=device, dtype=dtype)

        T_in = I_seq.shape[0]
        t_grid = torch.linspace(t_span[0], t_span[1], steps=T_in, dtype=dtype, device=device)

        def get_input(t: torch.Tensor) -> torch.Tensor:
            idx = torch.searchsorted(t_grid, t).item()
            idx = max(0, min(idx, T_in - 1))
            return I_seq[idx]

        def ode_func(t, x):
            I = get_input(t)
            return self.cell.ltc_dynamics(x, I)

        sol = odeint(
        ode_func,
        x0,
        t_save,
        method="rk4",   
)
        return sol  # (T_out, H)