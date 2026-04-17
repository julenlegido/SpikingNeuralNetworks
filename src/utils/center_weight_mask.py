import torch


def create_center_weight_mask(size=28, sigma=7.0):
    """
    Creates a 2D Gaussian mask centered in the image.
    """

    coords = torch.arange(size)
    x = coords.view(1, -1).repeat(size, 1)
    y = coords.view(-1, 1).repeat(1, size)

    center = (size - 1) / 2

    mask = torch.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))

    # normalize to [0,1]
    mask = mask / mask.max()

    return mask