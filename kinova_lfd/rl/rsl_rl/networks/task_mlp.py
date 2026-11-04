"""task_mlp.py

Supports a hierarchical RL framework of MLPs
"""

import torch
import torch.nn as nn

from kinova_lfd.rl.rsl_rl.networks import MLP


class TaskMLP(nn.Module):
    """Multi-layer perceptron.

    The MLP network is a sequence of linear layers and activation functions. The last layer is a linear layer that
    outputs the desired dimension unless the last activation function is specified.

    It provides additional conveniences:
    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        num_skills: int,
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_skills = num_skills
        self.output_dim = output_dim
        self.param_networks = nn.ModuleList(
            [
                MLP(
                    input_dim,
                    output_dim - num_skills,
                    hidden_dims,
                    activation=activation,
                    last_activation=last_activation,
                )
                for _ in range(num_skills)
            ]
        )

        self.task_network = MLP(
            input_dim, num_skills, hidden_dims, activation=activation, last_activation=last_activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.task_network(x)
        idx = logits.argmax(dim=-1)

        out = torch.zeros(size=(x.shape[0], self.output_dim), device=x.device)
        out[:, : self.num_skills] = logits
        for k, net in enumerate(self.param_networks):
            mask = idx == k
            if mask.any():
                out[mask, self.num_skills :] = net(x[mask])
        """for k in range(self.num_skills):
            mask = idx == k
            if mask.any():
                out[mask, self.num_skills :] = self.param_networks[k](x[mask])"""

        return out
