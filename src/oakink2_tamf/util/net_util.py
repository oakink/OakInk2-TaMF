import torch


def clip_gradient(optimizer: torch.optim.Optimizer, max_norm: float, norm_type: float):
    """Clips gradients computed during backpropagation to avoid explosion of gradients.

    Args:
        optimizer (torch.optim.optimizer): optimizer with the gradients to be clipped
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            torch.nn.utils.clip_grad.clip_grad_norm_(param, max_norm, norm_type)
