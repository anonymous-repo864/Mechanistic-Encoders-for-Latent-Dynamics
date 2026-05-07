from __future__ import annotations

from typing import Iterable, Sequence

import torch


def make_u_to_y_jump(
    control_indices: Sequence[int] | torch.Tensor,
    obs_indices: Sequence[int] | torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    
    '''
    Make a jump matrix J so that we can map the bolus control into the observed state y. 
    '''

    c = torch.as_tensor(control_indices, device=device)
    o = torch.as_tensor(obs_indices, device=device)

    U = int(c.shape[0])
    P = int(o.shape[0])

    # map full-state index -> observed dim
    obs_pos = {int(o[p].item()): p for p in range(P)}

    J = torch.zeros((U, P), dtype=dtype, device=c.device)
    for j in range(U):
        full_idx = int(c[j].item())
        p = obs_pos.get(full_idx, None)
        if p is not None:
            J[j, p] = 1.0

    return J
