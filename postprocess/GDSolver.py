import math

import torch
from torch import nn, optim


def solve(module: nn.Module, *inputs, lr=1e-2, tol=1e-4, max_iter=10000, optimizer=None, stop_tol=None, stop_range=None,
          return_best=True, **kwargs):
    history = None
    if stop_range is not None:
        assert stop_tol is not None

    best_loss = None
    best_state_dict = None

    with torch.enable_grad():
        if optimizer is None:
            optimizer = optim.SGD(module.parameters(), lr=lr)
        for i in range(max_iter):
            result = module(*inputs, **kwargs)
            loss = torch.abs(result)  # 优化目标固定设为方程=0
            if return_best and (best_loss is None or loss < best_loss):
                best_loss = loss.clone()
                state_dict = module.state_dict()
                best_state_dict = {k: state_dict[k].clone() for k in state_dict}
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < tol:
                break
            if stop_range is not None:
                if history is None:
                    history = torch.ones(stop_range, device=result.device, dtype=torch.float32) * math.inf
                history[i % stop_range] = result.item()
                if history.max() - history.min() < stop_tol:
                    break

        if return_best:
            # 最优的state_dict读回模型，返回
            module.load_state_dict(best_state_dict)

        return module
