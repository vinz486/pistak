import pytest
from pistak.hardware import HardwareEvaluator

def test_evaluate_hardware():
    he = HardwareEvaluator()
    hw_super = {"ram_gb": 32, "ov_devices": ["CPU", "GPU", "NPU"]}
    score, tier = he.evaluate_hardware(hw_super)
    assert score >= 8
    assert "Super PC" in tier

    hw_potato = {"ram_gb": 4, "ov_devices": ["CPU"]}
    score, tier = he.evaluate_hardware(hw_potato)
    assert score < 3
    assert "Potato PC" in tier

def test_rate_and_sort_models():
    he = HardwareEvaluator()
    hw = {"ram_gb": 16, "ov_devices": ["CPU", "GPU", "NPU"]}
    models = he.rate_and_sort_models(hw)
    
    # Highest stars first
    assert models[0]["stars_num"] >= models[-1]["stars_num"]
    # Verify stars are calculated
    assert "stars_str" in models[0]
