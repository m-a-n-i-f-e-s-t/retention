## POWER FULL KERNEL ##
# Implements the power self-attention algorithm using CUDA kernels.

## IMPLEMENTATION ##
import torch

POWER_FULL_DOC = r"""
Compute symmetric power retention with optional chunking.

This function implements the symmetric power retention mechanism from [1]. It generalizes
linear transformers by using symmetric power embeddings, which provide better expressivity
while maintaining tractable state sizes.

For a sequence of queries $Q_i$, keys $K_i$, and values $V_i ∈ ℝ^d$, the attention mechanism
computes outputs $Y_i ∈ ℝ^d$ as:

$$Y_i = Norm(\sum_{j=1}^i A_{ij} V_j)$$

where $Norm$ is a parameter-free layer normalization as follows:

$$Norm(x) = \frac{x - \mu(x)}{\sigma(x)}$$

where $\mu(x)$ and $\sigma(x)$ are the mean and standard deviation of $x$ along the feature dimension.

The attention weights are computed as follows:

$$A_{ij} = \frac{\phi(Q_i)^\top \phi(K_j)}{\sum_{k=1}^i \phi(Q_i)^\top \phi(K_k)}$$

Here $\phi$ is the symmetric power embedding that maps vectors to their deg-th symmetric power.
For long sequences, we use an equivalent RNN formulation with states $S_i$ and $Z_i$:

$$Y_{i} = \frac{S_i \phi(Q_i)}{Z_i \phi(Q_i)} \qquad Z_i = Z_{i-1} + \phi(K_i)^T \qquad S_i = S_{i-1} + V_i \phi(K_i)^T$$

The state size for each head is $D(d+1)$ where $D = \binom{d+deg-1}{deg}$, providing massive
savings over full tensor products (e.g., 96% reduction for deg=4).

Args:
    Q: Query tensor of shape `(batch_size, seq_len, num_q_heads, head_dim)`.
    K: Key tensor of shape `(batch_size, seq_len, num_kv_heads, head_dim)`.
    V: Value tensor of shape `(batch_size, seq_len, num_kv_heads, head_dim)`.
    log_G: Optional log gating factors of shape `(batch_size, seq_len, num_kv_heads)`.
        When provided, applies multiplicative gating to attention weights.
    initial_state: Optional initial state for recurrent processing. Not implemented yet.
    deg: Power attention degree. Must be even. Higher values make attention more "focused".
        Common values are:
        * deg=2: 49% state size reduction, slightly worse than baseline
        * deg=4: 96% reduction, outperforms baseline
        * deg=6: 99.8% reduction, best performance but large state
    scale: Scale factor for attention weights. Defaults to 1.0.
    chunk_size: Size of chunks for processing long sequences.
        If None, uses O(n²) attention formulation.
        If set, uses O(n) RNN formulation with chunked computation.
    temporal: Whether to use temporal normalization. Disabling this can hurt learning performance but may run a bit faster.

Returns:
    torch.Tensor: Output tensor of shape `(batch_size, seq_len, num_q_heads, head_dim)`.

Note:
    - Input tensors must have matching dtypes (fp16, bf16, or fp32)
    - If provided, log_G must be float32
    - Sequence length must be evenly divisible by chunk size
    - num_q_heads must be a multiple of num_kv_heads (for multi-query attention)
    - deg must be even for the symmetric power formulation
    - State size per head is $D(d+1)$ where $D = \binom{d+deg-1}{deg}$

References:
    [1] J. Buckman, C. Gelada, and S. Zhang, "Symmetric Power Transformers." 
        Manifest AI, Aug. 15, 2024.
"""

def pad_to_chunk_multiple(func):
    def padded_func(Q, K, V, log_G=None, initial_state=None, return_final_state=False,
                    deg=2, scale=None, chunk_size=None, switch_over_seq_len=None):
        t = Q.shape[1]
        need_pading = chunk_size is not None and t % chunk_size != 0
        if need_pading:
            t_pad = chunk_size - t % chunk_size
            Q = torch.nn.functional.pad(Q, (0, 0, 0, 0, 0, t_pad), 'constant', 0.)
            K = torch.nn.functional.pad(K, (0, 0, 0, 0, 0, t_pad), 'constant', 0.)
            V = torch.nn.functional.pad(V, (0, 0, 0, 0, 0, t_pad), 'constant', 0.)
            if log_G is not None:
                log_G = torch.nn.functional.pad(log_G, (0, 0, 0, t_pad), 'constant', 0.) # pad log_G with log 1 = 0
        out = func(Q, K, V, log_G, initial_state, return_final_state, deg, scale, chunk_size, switch_over_seq_len)
        if not need_pading:
            return out
        if return_final_state:
            O, S_final = out
            return O[:, :-t_pad], S_final
        return out[:, :-t_pad]
    return padded_func

def make_power_retention(update_state_impl, query_state_impl, discumsum_impl, attention_impl):
    """ Create a power_retention function with the given implementations.
    """
    @pad_to_chunk_multiple
    def _power_retention(Q, K, V, log_G=None, initial_state=None, return_final_state=False,
                    deg=2, scale=None, chunk_size=None, switch_over_seq_len=None): # noqa: C901
        assert deg % 2 == 0, f'deg must be even: {deg=}'
        
        _update_state = update_state_impl
        _query_state = query_state_impl
        _discumsum = discumsum_impl
        _attention = attention_impl
        
        # Establish shapes and dtypes
        assert Q.dtype == K.dtype == V.dtype, f'dtypes of inputs must match: {Q.dtype=}, {K.dtype=}, {V.dtype=}'
        dtype = Q.dtype
        b, t, hq, d = Q.shape
        _, _, h, _ = K.shape
        assert hq % h == 0, f"Q heads must be a multiple of KV heads: {hq=} {h=}"
        qhead_ratio = hq // h
        if chunk_size is not None:
            c = chunk_size
            assert t % chunk_size == 0, f'{t=} not evenly divisible by {chunk_size=}'
            n = t // chunk_size
            switch_over_seq_len = switch_over_seq_len if switch_over_seq_len is not None else chunk_size
        else:
            c = t
            n = 1
        gating = log_G is not None
        if gating:
            log_G = log_G.to(torch.float32)

        if not scale:
            scale = 1.0 / d**0.5

        # --- Simple quadratic attention ---
        if switch_over_seq_len is None or t <= switch_over_seq_len:
            log_G_accum = log_G.cumsum(1) if log_G is not None else None
            out = _attention(Q, K, V, log_G_accum, deg, scale=scale, norm=True)
            if not return_final_state:
                return out
            else:
                if gating:
                    log_discount_weights = (log_G_accum.narrow(1, c-1, 1) - log_G_accum) / deg
                    cs_K = K * torch.exp(log_discount_weights).unsqueeze(-1).to(K.dtype)
                S_final = _update_state(cs_K.unsqueeze(1).contiguous(), V.unsqueeze(1).contiguous(), deg)[:,-1]
                if initial_state is not None:
                    S_final += initial_state
                return out, S_final

        # --- Reshape into chunks ---
        Q = Q.view(b, n, c, hq, d)
        K = K.view(b, n, c, h, d)
        V = V.view(b, n, c, h, d)    
        if gating:
            log_G = log_G.view(b, n, c, h)
            log_G_intrachunk_accum = log_G.cumsum(2)

        # --- Update State ---
        if gating:
            log_discount_weights = (log_G_intrachunk_accum.narrow(2, c-1, 1) - log_G_intrachunk_accum) / deg
            cs_K = K * torch.exp(log_discount_weights).unsqueeze(-1).to(K.dtype)
        else:
            cs_K = K
        S, s = _update_state(cs_K.contiguous(), V.contiguous(), deg)

        # TODO(sean): properly handle initial state gating
        if initial_state is not None:
            S = torch.cat([initial_state.unsqueeze(1), S], dim=1) # n + 1 chunks

        # --- Accumulate ---
        if gating:
            log_G_chunk_sum = log_G_intrachunk_accum[:,:,-1].contiguous()
        else:
            log_G_chunk_sum = torch.zeros(size=(b, n, h), device=Q.device, dtype=torch.float32)
        S = _discumsum(S, log_G_chunk_sum) # Note that this adds an empty chunk to the start of the sequence
        S = S.narrow(1, 0 if initial_state is None else 1, n)
        s = _discumsum(s, log_G_chunk_sum)
        s = s.narrow(1, 0, n)
        if return_final_state:
            final_state = S[:,-1]

        # --- Merge chunks for attention ---
        Q, K, V = map(lambda x: x.contiguous(), (Q, K, V))
        if gating:
            log_G_intrachunk_accum = log_G_intrachunk_accum.contiguous()

        # --- Compute attention ---
        attn_Y, l_attn, rowmax = _attention(Q.view(b*n, c, hq, d), K.view(b*n, c, h, d), V.view(b*n, c, h, d), log_G_intrachunk_accum.view(b*n, c, h) if gating else None, deg, scale=scale, norm=False)
        attn_Y, l_attn, rowmax = map(lambda x: x.view(b, n, *x.shape[1:]), (attn_Y, l_attn, rowmax)) # [b, n, c, hq ...]
        # --- Gate Query for Query State ---
        if gating:
            if qhead_ratio > 1:
                log_G_intrachunk_accum = log_G_intrachunk_accum.repeat_interleave(qhead_ratio, dim=3)
            Q = Q * torch.exp(log_G_intrachunk_accum / deg).unsqueeze(-1).to(Q.dtype)

        # --- Compute Query State ---
        Q, S, s, attn_Y, l_attn, rowmax = map(lambda x: x.contiguous(), (Q, S, s, attn_Y, l_attn, rowmax))
        Y = _query_state(Q, S, s, attn_Y, l_attn, rowmax, deg, scale, initial_state is None)

        # Epilogue
        out = Y.contiguous().view(b, t, hq, d).to(dtype)
        if return_final_state:
            return out, final_state
        else:
            return out

    _power_retention.__doc__ = POWER_FULL_DOC
    return _power_retention


def make_power_retention_fused(update_state_impl, query_state_impl, discumsum_impl, attention_impl):
    """ Create a power_retention function with the given implementations.
    """
    @pad_to_chunk_multiple
    def _power_retention_fused(Q, K, V, log_G=None, initial_state=None, return_final_state=False,
                    deg=2, scale=None, chunk_size=None, switch_over_seq_len=None): # noqa: C901
        # assert deg % 2 == 0, f'deg must be even: {deg=}'
        _update_state = update_state_impl
        _query_state = query_state_impl
        _discumsum = discumsum_impl
        _attention = attention_impl
        
        # Establish shapes and dtypes
        assert Q.dtype == K.dtype == V.dtype, 'dtypes of inputs must match'
        dtype = Q.dtype
        b, t, hq, d = Q.shape
        _, _, h, _ = K.shape
        switch_over_seq_len = chunk_size if switch_over_seq_len is None else switch_over_seq_len
        c = t if chunk_size is None else chunk_size
        n = 1 if chunk_size is None else t // chunk_size
        assert t % c == 0, f'{t=} not evenly divisible by {c=}'
        gating = log_G is not None
        scale = 1.0 / d**0.5 if scale is None else scale

        # --- Simple quadratic attention ---
        V = V.clone()
        V[..., 0] = 1. # First feature is reserved for normalization
        if switch_over_seq_len is None or t <= switch_over_seq_len:
            log_G_accum = log_G.cumsum(1) if log_G is not None else None
            return _attention(Q, K, V, log_G_accum, deg, scale=scale, norm=True)

        # --- Reshape into chunks ---
        Q = Q.view(b, n, c, hq, d)  
        K = K.view(b, n, c, h, d)
        V = V.view(b, n, c, h, d)
        r = hq // h
        if gating:
            log_G = log_G.view(b, n, c, h)
            log_G_intrachunk_accum = log_G.cumsum(2)

        # --- Update State ---
        if gating:
            log_discount_weights = (log_G_intrachunk_accum.narrow(2, c-1, 1) - log_G_intrachunk_accum) / deg
            cs_K = K * torch.exp(log_discount_weights).unsqueeze(-1).to(K.dtype)
        else:
            cs_K = K
        S = _update_state(cs_K.contiguous(), V.contiguous(), deg)

        # --- Accumulate ---
        if gating:
            log_G_chunk_sum = log_G_intrachunk_accum[:,:,-1].contiguous()
        else:
            log_G_chunk_sum = torch.zeros(size=(b, n, h), device=Q.device, dtype=torch.float32)
        S = _discumsum(S, log_G_chunk_sum) # Note that this adds an empty chunk to the start of the sequence
        S = S.narrow(1, 0, n)

        # --- Merge chunks for attention ---
        Q, K, V = map(lambda x: x.contiguous(), (Q, K, V))
        log_G_intrachunk_accum = log_G_intrachunk_accum.contiguous() if gating else None

        def make_flatbatch(x):
            return x.view(b*n, *x.shape[2:]) if x is not None else None

        # --- Compute attention ---
        attn_Y, l_attn, rowmax = _attention(make_flatbatch(Q), make_flatbatch(K), make_flatbatch(V), make_flatbatch(log_G_intrachunk_accum), deg, scale=scale, norm=False)
        attn_Y, l_attn, rowmax = map(lambda x: x.view(b, n, *x.shape[1:]), (attn_Y, l_attn, rowmax)) # [b, n, c, h ...]
        # --- Gate Query for Query State ---
        Q = Q * torch.exp(log_G_intrachunk_accum.repeat_interleave(r, dim=-1) / deg).unsqueeze(-1).to(Q.dtype) if gating else Q # type: ignore

        # --- Compute Query State ---
        Q, S, attn_Y, l_attn, rowmax = map(lambda x: x.contiguous(), (Q, S, attn_Y, l_attn, rowmax))
        Y = _query_state(Q, S, attn_Y, l_attn, rowmax, deg, scale, zero_initial_state=True)

        # Epilogue
        out = Y.contiguous().view(b, t, hq, d).to(dtype)
        return out

    _power_retention_fused.__doc__ = POWER_FULL_DOC
    return _power_retention_fused


def make_power_retention_inference(update_state_impl, query_state_impl, attention_impl, fused_normalizer=True):
    """ Create a power_retention inference function with the given implementations.
    """
    def _power_retention_inference(Q, K, V, log_G=None, initial_state=None, sum_of_keys=None,
                    deg=2, scale=None, switch_over_seq_len=512) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        r"""
        Implements an efficient inference kernel for power retention, mathematically expressed as:

        $$
        y = q ⋅ S + (\phi(q) ⋅ \phi(K)^T) ⋅ V \\
        S = \begin{cases}
            S + \phi(K)^T ⋅ V & \text{if } \text{seq\_len} = \text{chunk\_size} \\
            S & \text{otherwise}
        \end{cases} \\
        $$
        where $S \in \mathbb{R}^{D \times d}$ is the initial_state, $q \in \mathbb{R}^{1 \times d}$ is the query, $K \in \mathbb{R}^{seq_len \times d}$ is the key, $V \in \mathbb{R}^{seq_len \times d}$ is the value, and $\phi$ is the symmetric power embedding. An optional gating factor $log_G$ is exponentiated and applied to the attention scores and initial_state.
        

        Args:
            Q: Query tensor of shape `(batch_size, tq, num_q_heads, head_dim)`.
            K: Key tensor of shape `(batch_size, seq_len, num_kv_heads, head_dim)`.
            V: Value tensor of shape `(batch_size, seq_len, num_kv_heads, head_dim)`.
            log_G: Optional log gating factors of shape `(batch_size, seq_len, num_kv_heads)`.
            initial_state: Optional initial_state for recurrent processing, of shape `(batch_size, num_kv_heads, D, head_dim)`.
            sum_of_keys: Optional sum of keys of shape `(batch_size, num_kv_heads, D)`, must be provided if fused_normalizer is False and initial_state is provided.
            deg: Power attention degree.
            scale: Scale factor for attention weights. Defaults to 1.0.
            switch_over_seq_len: The sequence length above which a initial_state update is performed. If None, no initial_state update other than gating is performed.

        Returns:
            output: Output tensor of shape `(batch_size, 1, num_q_heads, head_dim)`.
            initial_state: Updated initial_state tensor of shape `(batch_size, num_kv_heads, D, head_dim)`. 
        """
        _update_state = update_state_impl
        _query_state = query_state_impl
        _attention = attention_impl
        
        # Establish shapes and dtypes
        has_cache = K is not None and V is not None
        assert has_cache or initial_state is not None, 'initial_state or cache must be provided for inference'
        assert not has_cache or Q.dtype == K.dtype == V.dtype, 'dtypes of inputs must match'
        assert not has_cache or Q.shape[0] == K.shape[0] == V.shape[0], 'batch sizes of inputs must match'
        assert not has_cache or K.shape[1] == V.shape[1], 'key and value must have the same seq_len'
        assert initial_state is None or not has_cache or initial_state.shape[0] == K.shape[0], 'initial_state must have a batch size of the same as the key'
        assert initial_state is None or not has_cache or initial_state.shape[1] == K.shape[2], 'initial_state must have the same number of kv heads as the key'
        assert sum_of_keys is None or not has_cache or not fused_normalizer, 'sum_of_keys must be provided if fused_normalizer is False'
        assert log_G is None or not has_cache or (log_G.shape[:3] == K.shape[:3]), 'log_gating must have a batch size, seq_len, and head count of the same as the key'

        # Defaults and constants
        b, tq, hq, d = Q.shape
        if K is not None:
            _, tk, hk, _, e = *K.shape, V.shape[-1]
        else:
            assert initial_state is not None, 'initial_state or cache must be provided for inference'
            assert sum_of_keys is not None or fused_normalizer, 'sum_of_keys must be provided for inference'
            tk, hk, e = 0, initial_state.shape[1], initial_state.shape[-1]
        scale = 1.0 / d**0.5 if scale is None else scale

        if has_cache and fused_normalizer:
            V = V.clone()
            V[..., 0] = 1. # First feature is reserved for normalization

        # --- Query Cache + Query State ---
        log_G_accum = log_G.cumsum(1) if log_G is not None else None
        r, w = hq // hk, 1
        if initial_state is None:
            Y = _attention(Q, K, V, log_G_accum, deg, scale=scale, norm=True) # [b, tq, hq, e]
        else:
            assert initial_state.shape[0] == b, 'initial_state must have a batch size of the same as the query' # type: ignore
            assert initial_state.shape[1] == hk, 'initial_state must have the same number of kv heads as the key' # type: ignore
            if tk > 0:
                attn_Y, l_attn, rowmax = _attention(Q, K, V, log_G_accum, deg, scale=scale, norm=False) # [b, tq, hq, e], [b, tq, hq], [b, tq, hq]
            else:
                attn_Y, l_attn, rowmax = None, None, None
            if log_G_accum is not None:
                Q = Q * torch.exp(log_G_accum.narrow(1, -1, 1) / deg).repeat_interleave(r, dim=-1).unsqueeze(-1).to(Q.dtype) # discount the query when querying the initial_state, since it's cheaper than discounting the initial_state
            if fused_normalizer:
                Y = _query_state(Q, initial_state, attn_Y, l_attn, rowmax, deg, scale, zero_initial_state=False) # [b, tq, hq, e]
            else:
                Y = _query_state(Q, initial_state, sum_of_keys, attn_Y, l_attn, rowmax, deg, scale, zero_initial_state=False) # [b, tq, hq, e]

        # --- Optionally Compress Cache into State ---
        if switch_over_seq_len is not None and tk >= switch_over_seq_len:
            if log_G_accum is not None:
                log_discount_weights = (log_G_accum.narrow(1, -1, 1) - log_G_accum) / deg
                K = (K * torch.exp(log_discount_weights).unsqueeze(-1)).to(K.dtype)
            if fused_normalizer:
                state_update, sum_of_keys_update = _update_state(K.contiguous(), V.contiguous(), deg), None
            else:
                state_update, sum_of_keys_update = _update_state(K.contiguous(), V.contiguous(), deg)
            if initial_state is None:
                new_state = state_update
                new_sum_of_keys = sum_of_keys_update
            elif log_G_accum is None:
                new_state = initial_state + state_update
                new_sum_of_keys = sum_of_keys + sum_of_keys_update if not fused_normalizer else None # type: ignore
            else:
                new_state = initial_state * torch.exp(log_G_accum[:, -1, :]).unsqueeze(-1).unsqueeze(-1).to(initial_state.dtype) + state_update
                new_sum_of_keys = sum_of_keys * torch.exp(log_G_accum[:, -1, :]).unsqueeze(-1).to(sum_of_keys.dtype) + sum_of_keys_update if not fused_normalizer else None # type: ignore
        else:
            new_state = initial_state
            new_sum_of_keys = sum_of_keys if not fused_normalizer else None

        if fused_normalizer:
            return Y, new_state # type: ignore
        else:
            return Y, new_state, new_sum_of_keys

    

    _power_retention_inference.__doc__ = POWER_FULL_DOC
    return _power_retention_inference
