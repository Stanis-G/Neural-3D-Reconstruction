import torch
from torch import nn
from torch.nn import functional as F


def pos_encoder(coord, L, legacy=False):
    frequencies = 2 ** torch.arange(L, device=coord.device, dtype=coord.dtype)
    angles = coord[..., None] * frequencies
    if legacy:
        angles = coord[..., None] * torch.pi * frequencies
    encoding = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
    return encoding if legacy else torch.cat([coord, encoding], dim=-1)


def volume_rendering(points, densities, colors, white_bkgd=False, return_weights=False):
    delta = torch.linalg.vector_norm(points[:, 1:] - points[:, :-1], dim=-1)
    delta = torch.cat([delta, torch.full_like(densities[:, :1], 1e10)], dim=-1)
    alpha = -torch.expm1(-densities * delta)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1] + 1e-10], dim=-1),
        dim=-1,
    )
    weights = transmittance * alpha
    rgb = (weights[..., None] * colors).sum(dim=-2)
    if white_bkgd:
        rgb = rgb + (1 - weights.sum(dim=-1, keepdim=True))
    return (rgb, weights) if return_weights else rgb


def sample_pdf(bins, weights, num_samples, randomized):
    """Draw depths by inverting a piecewise-constant CDF."""
    weights = weights + 1e-5
    cdf = torch.cumsum(weights / weights.sum(dim=-1, keepdim=True), dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf[:, :-1], torch.ones_like(cdf[:, :1])], dim=-1)
    if randomized:
        u = torch.rand((cdf.shape[0], num_samples), device=cdf.device, dtype=cdf.dtype)
    else:
        u = (torch.arange(num_samples, device=cdf.device, dtype=cdf.dtype) + 0.5) / num_samples
        u = u.expand(cdf.shape[0], -1).contiguous()
    upper = torch.searchsorted(cdf, u, right=True).clamp(1, cdf.shape[-1] - 1)
    lower = upper - 1
    cdf_lo, cdf_hi = cdf.gather(-1, lower), cdf.gather(-1, upper)
    bin_lo, bin_hi = bins.gather(-1, lower), bins.gather(-1, upper)
    fraction = (u - cdf_lo) / (cdf_hi - cdf_lo).clamp_min(torch.finfo(cdf.dtype).eps)
    return torch.lerp(bin_lo, bin_hi, fraction)


class RadianceField(nn.Module):
    def __init__(self, depth, width, color_width, injection_layer, L_coords, L_direction,
                 legacy, negative_slope):
        super().__init__()
        self.L_coords = L_coords
        self.L_direction = L_direction
        self.legacy = legacy
        # Number of completed layers before concatenation.
        self.skip_after = injection_layer - int(legacy)
        input_features = 6 * L_coords + (0 if legacy else 3)
        direction_features = 6 * L_direction + (0 if legacy else 3)

        self.activation = nn.LeakyReLU(negative_slope) if legacy else nn.ReLU()
        self.pos_mlp_input_layer = nn.Linear(input_features, width)
        self.pos_mlp_inner_layers = nn.ModuleList(
            nn.Linear(width + input_features if index == self.skip_after else width, width)
            for index in range(1, depth)
        )
        self.density_mlp = nn.Sequential(nn.Linear(width, 1), nn.Softplus() if legacy else nn.ReLU())
        if not legacy:
            # Start with a small positive density to give empty rays a gradient.
            nn.init.constant_(self.density_mlp[0].bias, 0.1)
        self.color_features = nn.Identity() if legacy else nn.Linear(width, width)
        self.color_mlp = nn.Sequential(
            nn.Linear(width + direction_features, color_width),
            nn.LeakyReLU(negative_slope) if legacy else nn.ReLU(),
            nn.Linear(color_width, 3),
            nn.Sigmoid(),
        )

    def forward(self, points, viewdirs):
        encoding = pos_encoder(points, self.L_coords, legacy=self.legacy)
        features = self.activation(self.pos_mlp_input_layer(encoding))
        for index, layer in enumerate(self.pos_mlp_inner_layers, start=1):
            if index == self.skip_after:
                features = torch.cat([features, encoding], dim=-1)
            features = self.activation(layer(features))
        densities = self.density_mlp(features).squeeze(-1)
        directions = pos_encoder(viewdirs, self.L_direction, legacy=self.legacy)
        directions = directions[:, None, :].expand(-1, points.shape[1], -1)
        colors = self.color_mlp(torch.cat([self.color_features(features), directions], dim=-1))
        return densities, colors


class NeRF(nn.Module):
    def __init__(
        self,
        num_pos_mlp_layers,
        num_pos_mlp_layer_neurons,
        num_color_mlp_layer_neurons,
        num_injection_layer,
        negative_slope=0.2,
        num_samples=64,
        near=2.0,
        far=6.0,
        L_coords=10,
        L_direction=4,
        num_importance=128,
        white_bkgd=True,
        scene_center=(0.0, 0.0, 0.0),
        scene_scale=1.0,
        legacy=False,
    ):
        super().__init__()
        if not 1 <= num_injection_layer < num_pos_mlp_layers:
            raise ValueError("The skip must connect to an existing hidden layer.")
        if not 0 <= near < far or scene_scale <= 0:
            raise ValueError("Expected 0 <= near < far and a positive scene_scale.")
        if num_samples < 2 or num_importance < 0 or (not legacy and num_importance and num_samples < 3):
            raise ValueError("Use at least two coarse samples, or three with hierarchical sampling.")

        self.num_samples = num_samples
        self.num_importance = 0 if legacy else num_importance
        self.near = near
        self.far = far
        self.white_bkgd = white_bkgd and not legacy
        self.legacy = legacy
        self.scene_scale = scene_scale
        self.register_buffer("scene_center", torch.tensor(scene_center, dtype=torch.float32), persistent=False)

        field_args = (
            num_pos_mlp_layers, num_pos_mlp_layer_neurons, num_color_mlp_layer_neurons,
            num_injection_layer, L_coords, L_direction, legacy, negative_slope,
        )
        self.coarse = RadianceField(*field_args)
        self.fine = RadianceField(*field_args) if self.num_importance else None

    def render_samples(self, field, origins, directions, viewdirs, depths):
        points = origins[:, None, :] + depths[..., None] * directions[:, None, :]
        densities, colors = field((points - self.scene_center) * self.scene_scale, viewdirs)
        return volume_rendering(points, densities, colors, self.white_bkgd, return_weights=True)

    def forward(self, ray_origin, ray_direction, *, return_coarse=False):
        depths = torch.linspace(self.near, self.far, self.num_samples,
                                device=ray_origin.device, dtype=ray_origin.dtype)
        depths = depths.expand(ray_origin.shape[0], -1)
        randomized = self.training and not self.legacy
        if randomized:
            midpoints = (depths[:, 1:] + depths[:, :-1]) * 0.5
            lower = torch.cat([depths[:, :1], midpoints], dim=-1)
            upper = torch.cat([midpoints, depths[:, -1:]], dim=-1)
            depths = torch.lerp(lower, upper, torch.rand_like(depths))

        viewdirs = ray_direction if self.legacy else F.normalize(ray_direction, dim=-1)
        coarse_rgb, weights = self.render_samples(self.coarse, ray_origin, ray_direction, viewdirs, depths)
        rgb = coarse_rgb
        if self.fine is not None:
            midpoints = (depths[:, 1:] + depths[:, :-1]) * 0.5
            # Boundary weights are excluded, especially the artificial terminal interval.
            extra_depths = sample_pdf(midpoints, weights[:, 1:-1].detach(), self.num_importance, randomized)
            fine_depths = torch.cat([depths, extra_depths], dim=-1).sort(dim=-1).values
            rgb, _ = self.render_samples(self.fine, ray_origin, ray_direction, viewdirs, fine_depths)
        if return_coarse:
            return rgb, coarse_rgb if self.fine is not None else None
        return rgb
