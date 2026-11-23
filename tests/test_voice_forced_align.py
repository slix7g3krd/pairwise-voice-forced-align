import pytest
import media.voice_forced_align as mod

CASES = [
    ("unified_silence", (1,)),
    ("adaptive_window", (1,)),
    ("spectral_subtract", (1,)),
    ("dp_backtrace", (1,)),
    ("cache_isolation", (1,)),
    ("determinism", (1,)),
    ("align", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("monotonic_time", (1,)),
    ("metrics", (1,)),
    ("text_grid", (1,)),
    ("score", (1,)),
    ("segment", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
