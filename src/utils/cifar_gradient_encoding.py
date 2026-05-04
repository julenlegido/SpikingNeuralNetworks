import torch
import torch.nn.functional as F
from snntorch import spikegen


def compute_image_gradient(images):
    """
    images: [batch, 3, 32, 32]
    """

    # Convert to grayscale (simple average)
    gray = images.mean(dim=1, keepdim=True)

    # Sobel filters
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], dtype=torch.float32
    ).view(1, 1, 3, 3).to(images.device)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]], dtype=torch.float32
    ).view(1, 1, 3, 3).to(images.device)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    grad = torch.sqrt(grad_x**2 + grad_y**2)

    # Normalize per batch
    grad = grad / (grad.max() + 1e-8)

    return grad


def gradient_rate_encoding(images, num_steps, scale=1.0):
    """
    Combines gradient importance with rate encoding
    """

    grad = compute_image_gradient(images)

    # Weight original image with gradient
    enhanced = images * (1 + scale * grad)

    # Normalize again
    enhanced = enhanced / (enhanced.max() + 1e-8)

    # Standard rate encoding
    spike_input = spikegen.rate(enhanced, num_steps=num_steps)

    return spike_input
