import torch
import numpy as np
from IPython.display import clear_output 


def foot_of_perpendicular(a, b, p):
    ab = b - a
    ap = p - a

    ab_ab = torch.sum(ab * ab, dim=1, keepdim=True)
    ab_ap = torch.sum(ab * ap, dim=1, keepdim=True)

    t = ab_ap / ab_ab

    return (a + t * ab, t)


def distance_combination(a, b, p, multiplier=1):
    
    # repeat a and b to match the size of p
    a = a.repeat(p.size(0), 1)
    b = b.repeat(p.size(0), 1)

    l, t = foot_of_perpendicular(a, b, p)

    lp = p - l
    la = a - l

    lp_norm = torch.norm(lp, dim=1)
    la_norm = torch.norm(la, dim=1)

    combined = lp_norm + la_norm * multiplier

    # check if l is on the line
    t = t.squeeze()
    combined[t < 0] = -1

    return combined

def find_path_cosine(start, end, references, output=False, multiplier=None, max_sim=1, gpu_id=0):
    """
    Finds the path between a start point and an end point using cosine similarity.

    Args:
        start (torch.Tensor or list or tuple): The start point.
        end (torch.Tensor or list or tuple): The end point.
        references (torch.Tensor or list or tuple): The reference points.
        output (bool, optional): Whether to print output during the path finding process. Defaults to False.
        multiplier (float, optional): Legacy attribute. Not used in the current implementation. Defaults to None.
        max_sim (float, optional): The maximum cosine similarity allowed for a point to be considered valid. Defaults to 1.
        gpu_id (int, optional): The ID of the GPU to use for computation. Defaults to 0.

    Returns:
        list: The indices of the reference points that form the path from the start point to the end point.
    """
   
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

    if multiplier is not None:
        print("WARNUNG: multiplier attribute is legacy")

    if not isinstance(start, torch.Tensor):
        start = torch.tensor(start)
    if not isinstance(end, torch.Tensor):
        end = torch.tensor(end)
    if not isinstance(references, torch.Tensor):
        references = torch.tensor(references)

    start = start.to(device)
    end = end.to(device)
    references = references.to(device)

    points = torch.cat([references, end.unsqueeze(0)], dim=0)
    points_idx = torch.arange(points.shape[0]).to(device)

    current = start
    path_idx = []

    for _ in range(references.shape[0]):
        
        c2r = torch.nn.functional.cosine_similarity(points, current, dim=1)
        r2e = torch.nn.functional.cosine_similarity(points, end, dim=1)
        c2e = torch.nn.functional.cosine_similarity(current, end, dim=0)

        score = c2r
        
        mask1 = c2r <= max_sim
        mask2 = r2e >= c2e

        mask = mask1 & mask2
        valid = score[mask]

        if len(valid) == 0:
            print(f'no valid points left - took {len(path_idx[:-1])} reference points')
            break

        best = torch.argmax(valid)
        points = points[mask]
        points_idx = points_idx[mask]

        path_idx.append(points_idx[best].item())
        current = points[best]

        points = torch.cat([points[:best], points[best+1:]], dim=0)
        points_idx = torch.cat([points_idx[:best], points_idx[best+1:]], dim=0)

        if output:
            print(f'took {len(path_idx)-1}, \treferences left {points.shape[0]}')
    
    return path_idx[:-1] # remove the end point


def find_path(start, end, references, output=False, multiplier=1):
    """
    Calculates a path from the start point to the end point using a list of reference points.

    Args:
        start (numpy.ndarray): The starting point of the path.
        end (numpy.ndarray): The end point of the path.
        references (numpy.ndarray): A list of reference points.
        output (bool, optional): Whether to print optional output for debugging. Defaults to False.
        multiplier (int, optional): A multiplier to weight the combination of the vector directions. Smaller values will result in a more direct path. Defaults to 1.

    Returns:
        list: a list of indices of the reference points.
    """
    # 
    if isinstance(start, torch.Tensor):
        start =start.to('cpu')
    if isinstance(end, torch.Tensor):
        end = end.to('cpu')
    if isinstance(references, torch.Tensor):
        references = references.to('cpu')

    if start.shape[0] != end.shape[0] or start.shape[0] != references.shape[1]:
        print('invalid input dimensions')
        print(start.shape, end.shape, references.shape)
        return (None, None)

    # make sure all inputs are numpy arrays 
    # (for compatibility with tensors as input)
    start = np.array(start)
    end = np.array(end)
    references = np.array(references)

    points = np.vstack([references, end])
    point_idx = np.arange(points.shape[0])

    current = start
    path_idx = []

    for _ in range(points.shape[0]):

        # calculate heuristics
        h = distance_combination(torch.tensor(current), torch.tensor(end), torch.tensor(points), multiplier=multiplier)

        # filter out invalid heuristics values
        # (negative values are outside the line segment)
        mask = h >= 0
        valid = h[mask]
        if valid.shape[0] == 0:
            # end if no valid points are left
            print(f'no valid points left - took {len(path_idx[:-1])} reference points')
            break 

        # get best point and remove irrelevant points from the list
        best = np.argmin(valid)
        points = points[mask]
        point_idx = point_idx[mask]
        
        # optional output for debugging
        if output:
            print(f'took {len(path_idx)}, \treferences left {valid.shape[0]}')

        # add best point to the path and set it as start point for the next iteration
        path_idx.append(point_idx[best])
        current = points[best]

        points = np.delete(points, best, axis=0)
        point_idx = np.delete(point_idx, best)
    
    return path_idx[:-1] # remove the end point because it is not a reference point


def search_alpha(start, end, references, fnc, goal=25, tolerance=5, max_iter=500):

    """
    Search for the optimal value of alpha that produces a path with a desired length.

    Parameters:
    - start: The starting point of the path.
    - end: The ending point of the path.
    - references: A list of reference points.
    - fnc: The function used to generate the path.
    - goal: The desired length of the path.
    - tolerance: The allowed deviation from the desired length.
    - max_iter: The maximum number of iterations to perform.

    Returns:
    - path: The generated path.
    - alpha: The optimal value of alpha.

    """


    alpha = 0.5

    step = 0.1
    last = ''

    for i in range(max_iter):
        path = fnc(start, end, references, output=False, multiplier=alpha)
        l = len(path)
        if goal-tolerance <= l <= goal+tolerance:
            break
        elif l < goal-tolerance:
            if last == 'down':
                step /= 2
            alpha += step
            last = 'up'
        else:
            if last == 'up':
                step /= 2
            alpha -= step
            last = 'down'
        # print alpha 4 digits
        print(f'Alpha: {alpha:.4f} \t Length: {l}')
    
    print(f'Final alpha: {alpha:.4f} \t Length: {len(path)}')
    print()
    return path, alpha
        

