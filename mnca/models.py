import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MLP(nn.Module):
    """Multi-layer perceptron"""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, width: int):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, width))
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


class SHGNN(MessagePassing):
    """
    Spherical Harmonic Graph Neural Network

    Takes in a model that operates on each node and the spherical harmonics of its neighbors.

    s_i^{t+1} = s_i^t + model(s_i^t, z_i^t)


    Args:
        MessagePassing (_type_): _description_
    """

    def __init__(self, model: nn.Module, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.model = model

    @staticmethod
    def spherical_harmonics(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Returns the first 4 spherical harmonics evaluated at the given angles.

        Y_0_0 = 1 / (2 * sqrt(pi))
        Y_1_-1 = 1 / 2 * sqrt(3 / (2 * pi)) * sin(theta) * sin(phi)
        Y_1_0 = 1 / 2 * sqrt(3 / pi) * cos(theta)
        Y_1_1 = -1 / 2 * sqrt(3 / (2 * pi)) * sin(theta) * cos(phi)

        Returns [Y_0_0, Y_1_-1, Y_1_0, Y_1_1]

        Args:
            theta (torch.Tensor): _description_
            phi (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # TODO prealloc
        y_0_0 = 1 / (2 * torch.sqrt(torch.pi)) * torch.ones_like(theta)
        y_1_n1 = (
            1 / 2 * torch.sqrt(3 / (2 * torch.pi)) * torch.sin(theta) * torch.sin(phi)
        )
        y_1_0 = 1 / 2 * torch.sqrt(3 / torch.pi) * torch.cos(theta)
        y_1_1 = (
            -1 / 2 * torch.sqrt(3 / (2 * torch.pi)) * torch.sin(theta) * torch.cos(phi)
        )

        return torch.stack([y_0_0, y_1_n1, y_1_0, y_1_1], dim=-1)

    def message(
        self,
        s_i: torch.Tensor,
        s_j: torch.Tensor,
        angles_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the contribution of each neighbor to the central node. The contribution rule is based on
        the spherical harmonics of the angle between the central node and the neighbor node. Specifically,
        the contribution is w_{ij}^l * (s_j - s_i), where w_{ij}^l is the lth spherical harmonic of the angle
        from the ith point to the jth point. We use the first 4 spherical harmonics, Y_0^0, Y_1^{-1}, Y_1^0,
        and Y_1^1.

        Args:
            s_i (torch.Tensor): Central node features
            s_j (torch.Tensor): Neighbor node features
            angles_j (torch.Tensor): Spherical angles from central node to neighbor node

        Returns:
            torch.Tensor: w_{ij}^l * (s_j - s_i) for each spherical harmonic weight l
        """
        theta, phi = angles_j
        harmonics = SHGNN.spherical_harmonics(theta, phi)  # [num_edges, 4]

        # Broadcast to get the weighted difference w_{ij} * (s_j - s_i) for each channel
        # [num_edges, 4] * [num_edges, 1] -> [num_edges, 4]
        return (s_j - s_i) * harmonics

    def forward(
        self,
        s: torch.Tensor,
        h: torch.Tensor | None,
        angles: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the forward pass of the spherical harmonic graph neural network by performing message
        passing to get the perception vector, then passing that information through the specified
        model along with the feature vector and the optional conditional information vector.

        Args:
            s (torch.Tensor): Feature vector of the nodes
            h (torch.Tensor): Optional conditional information of the nodes
            angles (torch.Tensor): Spherical angles theta, phi from central nodes to their neighbors
            edge_index (torch.Tensor): Connectivity of the graph

        Returns:
            torch.Tensor: _description_
        """
        z = self.propagate(edge_index, s=s, angles=angles)

        match h:
            case None:
                v = torch.stack([s, z], dim=-1)
            case torch.Tensor():
                v = torch.stack([s, z, h], dim=-1)

        return s + self.model(v)
