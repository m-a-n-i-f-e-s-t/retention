import torch

def discumsum_reference(X, log_G):
    if log_G is None:
        return X.cumsum(axis=1, dtype=X.dtype)
    else:
        _, n, _ = log_G.shape
        chunk_G = torch.exp(log_G)
        chunk_G = chunk_G[...,None,None] if X.ndim == 5 else chunk_G[...,None]
        state = X[:, 0].to(torch.float32) + 0 * chunk_G[:, 0]
        state_stack = [torch.zeros_like(state).to(X.dtype), state.to(X.dtype)]
        for i in range(n-1):
            state = (state * chunk_G[:, i + 1]).to(X.dtype) + X[:, i + 1]
            state_stack.append(state)
        return torch.stack(state_stack, dim=1)
