import torch


def random_sampler(image_tensor, num_samples, generator, **kwargs):
    num_pixels = image_tensor.shape[1] * image_tensor.shape[2]

    return torch.randperm(
        num_pixels,
        generator=generator,
    )[:num_samples]


def gaussian_sampler(image_tensor, num_samples, generator, **kwargs):
    """Sample unique pixels using a Gaussian distribution centered on the image"""

    _, height, width = image_tensor.shape

    y, x = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij",
    )

    center_x = (width - 1) / 2
    center_y = (height - 1) / 2

    sigma_x = width / 4
    sigma_y = height / 4

    weights = torch.exp(
        -(
            ((x - center_x) ** 2) / (2 * sigma_x**2)
            + ((y - center_y) ** 2) / (2 * sigma_y**2)
        )
    )

    weights = weights.flatten()
    weights /= weights.sum()

    return torch.multinomial(
        weights,
        num_samples=num_samples,
        replacement=False,
        generator=generator,
    )


def foreground_sampler(
    image_tensor,
    num_samples,
    generator,
    transparent_fraction=0.2,
    **kwargs,
):
    alpha = image_tensor[3].flatten()

    transparent = torch.where(alpha == 0)[0]
    foreground = torch.where(alpha > 0)[0]

    num_transparent = round(num_samples * transparent_fraction)
    num_foreground = num_samples - num_transparent

    transparent = transparent[
        torch.randperm(
            len(transparent),
            generator=generator,
        )[:num_transparent]
    ]

    foreground = foreground[
        torch.randperm(
            len(foreground),
            generator=generator,
        )[:num_foreground]
    ]

    indices = torch.cat([transparent, foreground])

    return indices[
        torch.randperm(
            len(indices),
            generator=generator,
        )
    ]


SAMPLERS = {
    'random': random_sampler,
    'gaussian': gaussian_sampler,
    'foreground': foreground_sampler,
}


def get_sampler(name):
    if name not in SAMPLERS:
        raise ValueError(
            f"Unknown sampler '{name}'. "
            f"Available samplers: {list(SAMPLERS)}"
        )

    return SAMPLERS[name]
