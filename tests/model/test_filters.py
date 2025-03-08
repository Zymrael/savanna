import pytest 

from savanna.model.operators.hyena.parametrization.implicit_complex import ParallelComplexModalFilter

@pytest.mark.parametrize("L", [1024])
def test_parallel_complex_modal_filter_fwd(L):
    filter_fn = ParallelComplexModalFilter(d_model=32)
    h = filter_fn(L)
    print(h[0].shape)
    assert h is not None

# L = 1024
# filter_fn = ParallelComplexModalFilter(d_model=32)
# h = filter_fn(L)
# print(h[0].shape)
# assert h is not None