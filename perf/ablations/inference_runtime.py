import torch
from retention import power_retention
from retention.vidrial_fused import power_retention_inference
from retention.create_inputs import create_inference_inputs
from retention.vidrial_fused import update_state, query_state, attention
from fla.layers.gla import GatedLinearAttention
from fla.layers.rwkv7 import RWKV7Attention
from perf._timing import estimate_runtime, get_compiled_version, sanitize_kwargs
from perf.ablations.model import PowerAttention
from vidrial.py_utils.common import default_d_tile
from vidrial.kernels.sympow_mma.dimensions import sympow_dim
from vidrial.jit.settings import settings, PickBest
import pandas as pd
import logging
import gc
import torch.nn as nn


def flops_estimate(b, t, h, d, qhead_ratio, deg, chunk_size):
    D = sympow_dim(d, deg, d_tile=default_d_tile(d, deg))
    attention_flops = (qhead_ratio * t * d * 2 + qhead_ratio * d * t * 2) * b * h
    query_state_flops = (qhead_ratio * D * d * 2) * b * h
    update_state_flops = 0 if t < chunk_size else (D * d * 2 *t)
    return attention_flops + query_state_flops + update_state_flops

def measure_attention_time(**kwargs):
    t = kwargs['t']
    kwargs['t'] = kwargs['t'] % (kwargs['switch_over_seq_len'] + 1) if kwargs['t'] < (kwargs['switch_over_seq_len'] + 1) else (kwargs['t'] % (kwargs['chunk_size'] + 1))
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': t > kwargs['switch_over_seq_len'], 'device': 'cuda'})
    Q, K, V, log_G, state = inputs['Q'], inputs['K'], inputs['V'], inputs['log_G'], inputs['state']
    hq, hk = Q.shape[2], K.shape[2]
    log_G_accum = log_G.cumsum(1) if log_G is not None else None
    r, w = hq // hk, 1
    if kwargs.get('profile', False):
        sanitize_kwargs(attention)(**{**inputs, 'log_G_accum': log_G_accum, 'scale': 1.0 / kwargs['d']**0.5, 'norm': False})
        return 0
    else:
        time = estimate_runtime(get_compiled_version(attention, {**inputs, 'log_G_accum': log_G_accum, 'scale': 1.0 / kwargs['d']**0.5, 'norm': False}, direction='fwd', compile=False))
        return time

def measure_query_state_time(**kwargs):
    t, d, deg = kwargs['t'], kwargs['d'], kwargs['deg']
    kwargs['t'] = kwargs['t'] % (kwargs['switch_over_seq_len'] + 1) if kwargs['t'] < (kwargs['switch_over_seq_len'] + 1) else (kwargs['t'] % (kwargs['chunk_size'] + 1))
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': t > kwargs['switch_over_seq_len'], 'device': 'cuda'})
    if inputs['state'] is None:
        return 0
    Q, K, V, log_G, state, scale = inputs['Q'], inputs['K'], inputs['V'], inputs['log_G'], inputs['state'], inputs['scale']
    hq, hk = Q.shape[2], K.shape[2]
    log_G_accum = log_G.cumsum(1) if log_G is not None else None
    r, w = hq // hk, 1
    attn_Y, l_attn, rowmax = attention(Q, K, V, log_G_accum, deg, scale=scale, norm=False) # type: ignore
    if kwargs.get('profile', False):
        sanitize_kwargs(query_state)(**{**inputs, 'Y_attn': attn_Y, 'l_attn': l_attn, 'rowmax': rowmax, 'zero_initial_state': False, 'S': state})
        return 0
    else:
        time = estimate_runtime(get_compiled_version(query_state, {**inputs, 'Y_attn': attn_Y, 'l_attn': l_attn, 'rowmax': rowmax, 'zero_initial_state': False, 'S': state}, direction='fwd', compile=False))
        return time

def measure_update_state_time(**kwargs):
    t, chunk_size = kwargs['t'], kwargs['chunk_size']
    kwargs['t'] = kwargs['t'] % (kwargs['switch_over_seq_len'] + 1) if kwargs['t'] < (kwargs['switch_over_seq_len'] + 1) else (kwargs['t'] % (kwargs['chunk_size'] + 1))
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': t > kwargs['switch_over_seq_len'], 'device': 'cuda'})
    if kwargs.get('profile', False):
        sanitize_kwargs(update_state)(**inputs)
        return 0
    else:
        time = estimate_runtime(get_compiled_version(update_state, {**inputs}, direction='fwd', compile=False)) / chunk_size
        return time

def measure_total_time(**kwargs):
    t, chunk_size = kwargs['t'], kwargs['chunk_size']
    kwargs['t'] = kwargs['t'] % (kwargs['switch_over_seq_len'] + 1) if kwargs['t'] < (kwargs['switch_over_seq_len'] + 1) else (kwargs['t'] % (kwargs['chunk_size'] + 1))
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': t > kwargs['switch_over_seq_len'], 'device': 'cuda'})
    if kwargs.get('profile', False):
        sanitize_kwargs(power_retention_inference)(**inputs)
        return 0
    else:
        time = estimate_runtime(get_compiled_version(power_retention_inference, inputs, direction='fwd', compile=False))
        if kwargs['t'] > kwargs['switch_over_seq_len'] and kwargs['t'] % kwargs['chunk_size'] == 0:
            time -= measure_update_state_time(**kwargs) * (chunk_size - 1)
        torch.cuda.synchronize()
        gc.collect()
        return time

def measure_time(**kwargs):
    return {
        'total_time': measure_total_time(**kwargs),
        'attention_time': measure_attention_time(**kwargs),
        'query_state_time': measure_query_state_time(**kwargs),
        'update_state_time': measure_update_state_time(**kwargs),
    }


def measure_flashinfer_time(b, t, h, qhead_ratio, d, chunk_size, deg, gating, dtype, page_size=16, workspace_buffer=1024 * 1024 * 1024 * 4, **kwargs):
    import flashinfer
    workspace_buffer = torch.empty(workspace_buffer, dtype=torch.uint8, device="cuda:0")
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    pages_per_batch = t // page_size
    max_num_pages = b * pages_per_batch
    kv_page_indptr = torch.tensor(
        list(range(0, b * pages_per_batch, pages_per_batch)), dtype=torch.int32, device="cuda:0"
    )
    kv_page_indices = torch.arange(b * pages_per_batch).int().to("cuda:0")
    kv_last_page_len = torch.tensor(
        [page_size - 1] * b, dtype=torch.int32, device="cuda:0"
    )
    decode_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads=h * qhead_ratio,
        num_kv_heads=h,
        head_dim=d,
        page_size=page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    q = torch.randn(b, h * qhead_ratio, d, device="cuda:0", dtype=torch.bfloat16)
    kv_cache = torch.randn(
        max_num_pages, 2, page_size, h, d, device="cuda:0", dtype=torch.bfloat16
    )
    time = estimate_runtime(get_compiled_version(decode_wrapper.run, {'q': q, 'paged_kv_cache': kv_cache}, direction='fwd', compile=False), num1=2, num2=10)
    torch.cuda.synchronize()
    gc.collect()
    return time

def measure_rwkv7_time(b, t, h, qhead_ratio, d, chunk_size, deg, gating, dtype, **kwargs):
    h = h * qhead_ratio # rwkv7 doesn't support GQA, so we have to bump up the total number of heads
    hidden_size = h * d
    head_dim = d
    model = RWKV7Attention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=h,
        layer_idx=0,
        num_hidden_layers=2, 
        **kwargs
    ).to('cuda').to(dtype)
    model.eval()
    hidden_states = torch.randn(b, 1, hidden_size, device="cuda", dtype=dtype)
    past_key_value = {0:{
        'conv_state': torch.randn(b, hidden_size, device="cuda", dtype=dtype),
        'recurrent_state': torch.randn(b, h, d, d, device="cuda", dtype=dtype),
    }}
    time = estimate_runtime(get_compiled_version(model.forward, {'hidden_states': hidden_states, 'past_key_values': past_key_value, 'use_cache': True, 'output_attentions': False, 'v_first': None, 'cu_seqlens': None}, direction='fwd', compile=False), num1=2, num2=10)
    torch.cuda.synchronize()
    gc.collect()
    return time

def measure_gla_time(b, t, h, qhead_ratio, d, chunk_size, deg, gating, dtype, **kwargs):
    model = GatedLinearAttention(
        mode='fused_recurrent',
        hidden_size=h * d * qhead_ratio,
        expand_k=1.0,
        expand_v=1.0,
        num_heads=h*qhead_ratio,
        num_kv_heads=h,
        layer_idx=0,
        **kwargs
    ).to('cuda').to(dtype)
    model.eval()
    hidden_states = torch.randn(b, 1, h * d * qhead_ratio, device="cuda", dtype=dtype)
    past_key_value = {0:{
        'conv_state': torch.randn(b, h * d, device="cuda", dtype=dtype),
        'recurrent_state': torch.randn(b, h, d, d, device="cuda", dtype=dtype),
    }}
    time = estimate_runtime(get_compiled_version(model.forward, {'hidden_states': hidden_states, 'past_key_values': past_key_value, 'use_cache': True}, direction='fwd', compile=False), num1=2, num2=10)
    torch.cuda.synchronize()
    gc.collect()
    return time

def measure_power_model_time(b, t, h, qhead_ratio, d, chunk_size, deg, gating, dtype, **kwargs):
    t = t % (kwargs['switch_over_seq_len'] + 1) if t < (kwargs['switch_over_seq_len'] + 1) else (t % (chunk_size + 1))
    model = PowerAttention(
        num_heads=h,
        hidden_size=h * d,
        chunk_size=chunk_size,
        degree=deg,
        head_size=d,
        qhead_ratio=qhead_ratio,
        gating=gating,
        kernel='power',
        device='cuda',
        dtype=dtype,
    ).to('cuda').to(dtype)
    D = sympow_dim(d, deg, d_tile=default_d_tile(d, deg))
    model.eval()
    hidden_states = torch.randn(b, 1, h * d, device="cuda", dtype=dtype)
    past_key_values = {0:{
        'k': torch.randn(b, t, h, d, device="cuda", dtype=dtype),
        'v': torch.randn(b, t, h, d, device="cuda", dtype=dtype),
        'state': torch.randn(b, h, D, d, device="cuda", dtype=dtype),
        'g': torch.randn(b, t, h, device="cuda", dtype=dtype),
    }}
    time = estimate_runtime(get_compiled_version(lambda hidden_states, past_key_values: model(hidden_states=hidden_states, past_key_values=past_key_values, use_cache=True)[0], {'hidden_states': hidden_states, 'past_key_values': past_key_values}, direction='fwd', compile=False), num1=2, num2=10)
    torch.cuda.synchronize()
    gc.collect()
    return time

def inference_time_breakdown(profile=False):
    df = []
    b, t, h, d, chunk_size, deg, gating, dtype = 32, 64, 8, 64, 64, 2, True, torch.bfloat16
    print(f"Measuring runtime for {b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}")
    logging.basicConfig(level=logging.ERROR)

    with settings.set(policy=PickBest):
        for qhead_ratio in [1, 8]:
            print(f"========== {qhead_ratio=} ==========")
            df.append({
                **measure_time(b=b, t=t%chunk_size, h=h, d=d, qhead_ratio=qhead_ratio, deg=deg, chunk_size=chunk_size, dtype=dtype, gating=gating),
                'group': qhead_ratio,
            })

    df = pd.DataFrame(df)
    print(df)


    import matplotlib.pyplot as plt

    # Create stack plot
    plt.figure(figsize=(12, 8))

    # Prepare data for stack plot
    x = df['group']
    attention_times = df['attention_time']
    query_state_times = df['query_state_time'] 
    update_state_times = df['update_state_time']

    # Create the stack plot
    plt.stackplot(x, attention_times, query_state_times, update_state_times,
                labels=['Attention', 'Query State', 'Update State'],
                alpha=0.8)

    # Also plot the total time as a line for reference
    plt.plot(x, df['total_time'], 'k--', linewidth=2, marker='o', label='Total Time', markersize=6)

    plt.xlabel('Group Size (qhead_ratio)')
    plt.ylabel('Time (ms)')
    plt.title(f'Inference Time Breakdown vs Group Size\n{b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'power_inference_time_breakdown_{b}_{t}_{h}_{d}_{chunk_size}_{deg}_{gating}_{dtype}.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_with_flashinfer():
    # install from https://github.com/flashinfer-ai/flashinfer
    df = []
    b, h, d, chunk_size, deg, gating, dtype, switch_over_seq_len = 32, 8, 64, 64, 2, True, torch.bfloat16, 512
    print(f"Measuring runtime for {b=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}")
    logging.basicConfig(level=logging.ERROR)
    qhead_ratio = 8

    with settings.set(policy=PickBest):
        for t in [128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            print(f"========== {t=} ==========")
            power_measurement = measure_power_model_time(b=b, t=t, h=h, qhead_ratio=qhead_ratio, d=d, chunk_size=chunk_size, deg=deg, gating=gating, dtype=dtype, switch_over_seq_len=switch_over_seq_len)
            gla_measurement = measure_gla_time(b=b, t=t, h=h, qhead_ratio=qhead_ratio, d=d, chunk_size=chunk_size, deg=deg, gating=gating, dtype=dtype)
            rwkv7_measurement = measure_rwkv7_time(b=b, t=t, h=h, qhead_ratio=qhead_ratio, d=d, chunk_size=chunk_size, deg=deg, gating=gating, dtype=dtype)
            # power_measurement = measure_total_time(b=b, t=t, h=h, d=d, qhead_ratio=qhead_ratio, deg=deg, chunk_size=chunk_size, dtype=dtype, gating=gating, switch_over_seq_len=switch_over_seq_len)
            flash_measurement = measure_flashinfer_time(b=b, t=t, h=h, qhead_ratio=qhead_ratio, d=d, chunk_size=chunk_size, deg=deg, gating=gating, dtype=dtype)
            df.append({
                'Context': t,
                'FlashInfer (ms)': flash_measurement,
                'RWKV7 (ms)': rwkv7_measurement,
                'GLA (ms)': gla_measurement,
                'Power (ms)': power_measurement,
                'RWKV7 Speedup': flash_measurement / rwkv7_measurement,
                'GLA Speedup': flash_measurement / gla_measurement,
                'Power Speedup': flash_measurement / power_measurement,
            })

    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    df = pd.DataFrame(df)
    print("\nFull results table:")
    print(df.to_string())
    
    # Create pivoted table
    pivoted_df = df.set_index('Context')[['FlashInfer (ms)', 'RWKV7 Speedup', 'GLA Speedup', 'Power Speedup']].T
    print("\nPivoted table (speedup over flashinfer):")
    print(pivoted_df.to_string())
    
    # Print raw data
    print("\nRaw data:")
    print(df.to_dict('records'))



    # import matplotlib.pyplot as plt

    # # Create stack plot
    # plt.figure(figsize=(12, 8))

    # # Prepare data for stack plot
    # x = df['Context']
    # power_inference_time = df['Power (ms)']
    # flashinfer_time = df['FlashInfer (ms)']
    # rwkv7_time = df['RWKV7 (ms)']
    # gla_time = df['GLA (ms)']
    # query_state_time = df['query_state_time']
    # update_state_time = df['update_state_time']


    # Create the stack plot
    # plt.plot(x, power_inference_time, 'k--', color='red', linewidth=2, marker='o', label='Power Attention', markersize=6)
    # plt.plot(x, flashinfer_time, 'k-', color='blue', linewidth=2, marker='o', label='FlashInfer', markersize=6)
    # # plt.plot(x, attention_time, 'k-', color='green', linewidth=2, marker='o', label='Attention', markersize=6)
    # # plt.plot(x, query_state_time, 'k-', color='yellow', linewidth=2, marker='o', label='Query State', markersize=6)
    # # plt.plot(x, update_state_time, 'k-', color='purple', linewidth=2, marker='o', label='Update State', markersize=6)

    # plt.xlabel('Context Size (tokens)')
    # plt.ylabel('Time (ms)')
    # plt.title(f'Power Infer vs FlashInfer\n{b=} {h=} {qhead_ratio=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}')
    # plt.legend(loc='upper left')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()

    # # Save the plot
    # plt.savefig(f'power_infer_vs_flashinfer_{b}_{h}_{qhead_ratio}_{d}_{chunk_size}_{deg}_{gating}_{dtype}.png', dpi=150, bbox_inches='tight')
    # plt.show()


def torch_profile():
    from torch.profiler import profile, ProfilerActivity, record_function
    b, t, h, d, chunk_size, deg, gating, dtype = 32, 64, 8, 64, 64, 2, True, torch.bfloat16
    qhead_ratio = 16
    print(f"Profiling runtime for {b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=} {qhead_ratio=}")
    inputs = create_inference_inputs(b=b, t=t, h=h, d=d, qhead_ratio=qhead_ratio, dtype=dtype, device='cuda', gating=gating, chunk_size=chunk_size, deg=deg, initial_state=True)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True) as prof:
        power_retention_inference(inputs['Q'], inputs['K'], inputs['V'], inputs['log_G'], inputs['state'], deg=deg, scale=1.0 / d**0.5, chunk_size=chunk_size)
    prof.export_chrome_trace(f'power_inference_time_breakdown_{b}_{t}_{h}_{d}_{chunk_size}_{deg}_{gating}_{dtype}_{qhead_ratio}.json')

    print(prof.key_averages(group_by_stack_n=2).table(sort_by='cuda_time_total', row_limit=10))

if __name__ == '__main__':
    compare_with_flashinfer()
