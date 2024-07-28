from tqdm.notebook import tqdm_notebook as tqdm
import numpy as np
import torch


def interpolate_lerp(start_tensor, dest_tensor, n_steps):
    """
    Interpolates between two tensors using linear interpolation (lerp).

    Args:
        start_tensor (torch.Tensor): The starting tensor.
        dest_tensor (torch.Tensor): The destination tensor.
        n_steps (int): The number of interpolation steps.

    Returns:
        torch.Tensor: The interpolated tensor.
    """
    assert start_tensor.shape == dest_tensor.shape
    assert start_tensor.device == dest_tensor.device
    
    # Expand dimensions to match the target shape
    start_tensor = start_tensor.unsqueeze(0).unsqueeze(0)
    dest_tensor = dest_tensor.unsqueeze(0).unsqueeze(0)

    # Create a linear space tensor for steps
    step_values = torch.linspace(0, 1, n_steps, device=start_tensor.device)

    # Expand dimensions to allow broadcasting
    step_values = step_values.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    # Calculate the entire interpolation in a single operation
    interpolated_result = start_tensor + (dest_tensor - start_tensor) * step_values

    return interpolated_result.squeeze(0).squeeze(0)


def slerp(t, v0, v1, DOT_THRESHOLD=0.9999995):
    """
    Performs spherical linear interpolation (slerp) between two vectors.

    Args:
        t (float): The interpolation parameter.
        v0 (torch.Tensor): The starting vector.
        v1 (torch.Tensor): The destination vector.
        DOT_THRESHOLD (float, optional): The threshold for considering vectors as parallel. Defaults to 0.9999995.

    Returns:
        torch.Tensor: The interpolated vector.
        bool: True if linear interpolation was used, False otherwise.
    """
    v0_norm = v0 / torch.norm(v0, dim=-1, keepdim=True)
    v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True)
    dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)

    if dot.item() > DOT_THRESHOLD:
        # The vectors are almost parallel, use linear interpolation
        result = v0 + t * (v1 - v0)
        return result, True

    # angle between embeddings
    theta_0 = torch.acos(dot)

    # save the sin so it is not computed multiple times
    sin_theta_0 = torch.sin(theta_0)

    theta_t = theta_0 * t

    s0 = torch.sin((1-t) * theta_0) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0

    return s0 * v0 + s1 * v1, False


def interpolate_slerp(start_tensor, dest_tensor, n_steps):
    """
    Interpolates between two tensors using spherical linear interpolation (slerp).

    Args:
        start_tensor (torch.Tensor): The starting tensor.
        dest_tensor (torch.Tensor): The destination tensor.
        n_steps (int): The number of interpolation steps.

    Returns:
        torch.Tensor: The interpolated tensor.
    """
    assert start_tensor.shape == dest_tensor.shape, "Tensors must have the same shape"
    assert start_tensor.device == dest_tensor.device
    
    num_embeddings = start_tensor.shape[0]
    embedding_dim = start_tensor.shape[1]
    
    interpolated_tensors = torch.zeros((n_steps, num_embeddings, embedding_dim))

    count = 0
    for step in range(n_steps):
        t = step / (n_steps - 1)
        for i in range(num_embeddings):
            interpolated_tensors[step, i], used_lin = slerp(t, start_tensor[i], dest_tensor[i])
            if used_lin:
                count += 1
    print(f'info: used lerp for {int(count/n_steps)} out of {num_embeddings} embeddings')
    
    return interpolated_tensors.to(start_tensor.device)


def getNoise(batch_size, seed=0):
    """
    Generates random noise tensors.

    Args:
        batch_size (int): The number of noise tensors to generate.
        seed (int, optional): The random seed. Defaults to 0.

    Returns:
        torch.Tensor: The generated noise tensors.
    """
    t = torch.randn((4,64,64), generator=torch.manual_seed(seed))
    return t.repeat((batch_size, 1,1,1))