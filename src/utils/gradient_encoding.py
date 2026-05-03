import torch
import torch.nn.functional as F


def compute_gradient_map(images):
    """
    Compute gradient magnitude using Sobel filters.
    images: [B, 1, 28, 28]
    """

    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=torch.float32,
        device=images.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [0,  0,  0],
         [1,  2,  1]],
        dtype=torch.float32,
        device=images.device
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(images, sobel_x, padding=1)
    grad_y = F.conv2d(images, sobel_y, padding=1)

    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)

    return grad_mag


def normalize_gradient(grad):
    """
    Normalize gradient map to [0,1]
    """

    grad_min = grad.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    grad_max = grad.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

    grad_norm = (grad - grad_min) / (grad_max - grad_min + 1e-8)

    return grad_norm