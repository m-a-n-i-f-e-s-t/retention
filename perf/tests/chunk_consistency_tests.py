import torch
import pytest
from perf._checks import (
    check_inputs_forwards_match,
    check_inputs_backwards_match,
)
from perf.tests.test_list import power_retention_param_ranges, power_retention_input_output, fn_set_and_param_range_to_test_cases
from retention.vidrial_reference import power_retention as power_retention_vidrial_reference
from retention.reference import power_retention as power_retention_reference
power_retention_fn_sets = [
    {'name': 'power_retention_reference', 'extends': 'power_retention', 'impl': 'reference',
        'fn': power_retention_reference, **power_retention_input_output},
    {'name': 'power_retention_vidrial_reference', 'extends': 'power_retention', 'impl': 'vidrial_reference',
        'fn': power_retention_vidrial_reference, **power_retention_input_output},
]
TEST_CASES = fn_set_and_param_range_to_test_cases(power_retention_fn_sets, power_retention_param_ranges)

## CHUNK CONSISTENCY TESTS ##
# These tests confirm that the reference implementation is invariant to chunking

@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_consistency_wrt_chunk_size(fns_params):
    fns, params = fns_params
    if params['chunk_size'] is None: pytest.skip("Skipping test for chunk_size=None, because it is vacuously true")
    inputs_attention = fns['create_inputs'](**(params | {'chunk_size': None, 'dtype': torch.float32}))
    inputs_recurrent = fns['create_inputs'](**(params | {'dtype': torch.float32}))
    check_inputs_forwards_match(
        fn=fns['fn'],
        inputs1=inputs_attention,
        inputs2=inputs_recurrent,
        atol=1e-1,
    )

@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_grad_consistency_wrt_chunk_size(fns_params):
    fns, params = fns_params
    if params['chunk_size'] is None: pytest.skip("Skipping test for chunk_size=None, because it is vacuously true")
    inputs_attention = fns['create_inputs'](**(params | {'chunk_size': None, 'dtype': torch.float32}), requires_grad=True)
    inputs_recurrent = fns['create_inputs'](**(params | {'dtype': torch.float32}), requires_grad=True)
    check_inputs_backwards_match(
        fn=fns['fn'],
        inputs1=inputs_attention,
        inputs2=inputs_recurrent,
        atol=1e-2,
    )


# TODO(sean): find a better place for this test
# @pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
# def test_power_retention_reference_log_space_consistency(kw):
#     if kw['log_space'] is True: pytest.skip("Skipping test for log_space=True")
#     inputs_log_space = create_inputs(**(kw | {'log_space': True, 'dtype': torch.float32}))
#     inputs_normal_space = create_inputs(**(kw | {'log_space': False, 'dtype': torch.float32}))

#     check_inputs_forwards_match(
#         fn=power_retention_reference,
#         inputs1=inputs_log_space,
#         inputs2=inputs_normal_space,
#         atol=1e-1,
#     )

# @pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
# def test_power_retention_reference_log_space_grad_consistency(kw):
#     if kw['log_space'] is True: pytest.skip("Skipping test for log_space=True")
#     inputs_log_space = create_inputs(**(kw | {'log_space': True, 'dtype': torch.float32}), requires_grad=True)
#     inputs_normal_space = create_inputs(**(kw | {'log_space': False, 'dtype': torch.float32}), requires_grad=True)
#     check_inputs_backwards_match(
#         fn=power_retention_reference,
#         inputs1=inputs_log_space,
#         inputs2=inputs_normal_space,
#         atol=1e-3,
#     )
