import torch
from torch import nn

from dataset import sample_points_on_ray


def pos_encoder(coord, L):
    """Apply NeRF positional encoding to coordinates"""
    freq = 2 ** torch.arange(L, device=coord.device, dtype=coord.dtype)
    angles = coord[..., None] * torch.pi * freq

    encoding = torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=-1,
    )
    return encoding.flatten(-2)


def volume_rendering(points, densities, colors):
    """
    Sample color of a pixel along a ray

    Args:
        points: coordinates of sampled points along a ray, shape (N_samples, 3)
        densities: densities of points along a ray, shape (N_samples)
        colors: colors of points along a ray, shape (N_samples, 3)

    Returns:
        rgb color, shape (3)
    """
    # Calculate distances between neighboring points
    delta = points[:, 1:] - points[:, :-1] # (Batch, N_samples-1, 3)

    # Convert vectors into distances between points
    delta = torch.linalg.norm(delta, dim=-1) # (Batch, N_samples-1)

    # Add last interval to delta: a distance to infinite point
    last_delta = torch.full_like(delta[:, :1], 1e10)

    delta = torch.cat([delta, last_delta], dim=1) # (Batch, N_samples)

    # Probability of ray to terminate inside each interval
    alpha = 1.0 - torch.exp(-densities * delta) # (Batch, N_samples)

    # Calculate accumulated transparency along a ray
    # (probability of ray to travel from ray origin to current point without termination)
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones_like(alpha[:, :1]),
            1.0 - alpha[:, :-1] + 1e-10
        ], dim=1),
        dim=1
    ) # (Batch, N_samples)

    weights = transmittance * alpha # (Batch, N_samples)

    # Calculate pixel color
    rgb = torch.sum(weights[..., None] * colors, dim=1) # (Batch, 3)
    return rgb


class NeRF(nn.Module):

    def __init__(self,
            num_pos_mlp_layers,
            num_pos_mlp_layer_neurons,
            num_color_mlp_layer_neurons,
            num_injection_layer,
            negative_slope,
            num_samples,
            near,
            far,
            L_coords=10,
            L_direction=4,
        ):
        super().__init__()
        self.num_injection_layer = num_injection_layer
        self.num_samples = num_samples
        self.near = near
        self.far = far
        self.L_coords = L_coords
        self.L_direction = L_direction

        if not 1 <= num_injection_layer < num_pos_mlp_layers:
            raise ValueError(
                "num_injection_layer must be between 1 and "
                "num_pos_mlp_layers - 1"
            )

        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

        # Size of positional encoding of ray coords. 2 stands for sin and cos. 3 stands for each coord of a point
        input_features = self.L_coords * 2 * 3

        # Add input layer
        self.pos_mlp_input_layer = nn.Linear(
            in_features=input_features,
            out_features=num_pos_mlp_layer_neurons,
            bias=True,
        )

        # Add inner layers
        pos_mlp_inner_layers = []
        for layer_idx in range(1, num_pos_mlp_layers):
            # - 1 to account for input layer
            if layer_idx == num_injection_layer - 1:
                # Increase size of in_features a layer with injection
                in_features = (num_pos_mlp_layer_neurons + input_features)
            else:
                in_features = num_pos_mlp_layer_neurons
            pos_mlp_inner_layers += [
                nn.Linear(
                    in_features=in_features,
                    out_features=num_pos_mlp_layer_neurons,
                    bias=True,
                )
            ]

        self.pos_mlp_inner_layers = nn.ModuleList(pos_mlp_inner_layers)

        self.density_mlp = nn.Sequential(
            nn.Linear(
                in_features=num_pos_mlp_layer_neurons,
                out_features=1, # return 1 density per point
                bias=True,
            ),
            nn.Softplus(),
        )

        self.color_mlp = nn.Sequential(
            nn.Linear(
                # Size of positional encoding of ray coords + ray direction. 2 stands for sin and cos. 3 stands for each coord of a vector
                in_features=num_pos_mlp_layer_neurons + self.L_direction * 2 * 3,
                out_features=num_color_mlp_layer_neurons,
                bias=True,
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(
                in_features=num_color_mlp_layer_neurons,
                out_features=3,
                bias=True,
            ),
            nn.Sigmoid(),
        )


    def forward(self, ray_origin, ray_direction):
        points = sample_points_on_ray(
            ray_origin,
            ray_direction,
            near=self.near,
            far=self.far,
            num_samples=self.num_samples,
        )
        coords_encoding = pos_encoder(points, L=self.L_coords)

        # Run positional mlp input layer
        coords_feature_vector = self.activation(self.pos_mlp_input_layer(coords_encoding))

        # Run positional mlp inner layers
        for layer_idx, layer in enumerate(self.pos_mlp_inner_layers):
            # -1 to account for input layer (also consisting of linear and activation)
            # Another -1 to account for that indexing starts from 0
            if layer_idx == self.num_injection_layer - 2:
                # Inject positional encoding
                coords_feature_vector = torch.cat(
                    [coords_feature_vector, coords_encoding],
                    dim=-1,
                )
            coords_feature_vector = self.activation(layer(coords_feature_vector))

        # Output densities
        densities = self.density_mlp(coords_feature_vector).squeeze(-1)

        # Encode ray direction
        dir_encoding = pos_encoder(
            ray_direction[:, None, :].expand(-1, self.num_samples, -1),
            L=self.L_direction,
        )
        feature_vector = torch.cat([coords_feature_vector, dir_encoding], dim=-1)
        colors = self.color_mlp(feature_vector)

        pixel_color = volume_rendering(points, densities, colors)
        return pixel_color
