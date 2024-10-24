# pinn_model.py

import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

def pde_residual_anisotropic(model, x, k):
    x.requires_grad = True
    u = model(x)

    # Compute gradients
    grads = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, [0]]
    u_y = grads[:, [1]]

    # Compute gradient magnitude
    grad_u_norm = torch.sqrt(u_x**2 + u_y**2 + 1e-12)  # Add epsilon to avoid division by zero

    # Compute conductivity function c(|∇u|)
    c = torch.exp(-(grad_u_norm / k)**2)

    # Compute components of c * ∇u
    c_u_x = c * u_x
    c_u_y = c * u_y

    # Compute divergence of c * ∇u
    div_c_u_x = torch.autograd.grad(c_u_x, x, torch.ones_like(c_u_x), create_graph=True)[0][:, [0]]
    div_c_u_y = torch.autograd.grad(c_u_y, x, torch.ones_like(c_u_y), create_graph=True)[0][:, [1]]
    div_c_u = div_c_u_x + div_c_u_y

    # PDE residual (since ∂u/∂t = 0 in steady state)
    residual = div_c_u

    return residual
