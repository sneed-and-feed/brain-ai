"""
Tests for scripts/trm_colab_runner.py helpers that do not need a GPU or the TRM checkout:
the pure-PyTorch AdamATan2 fallback must reproduce the adam-atan2 0.0.3 reference update exactly.
"""

import importlib.util
import math
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_runner():
    spec = importlib.util.spec_from_file_location("trm_colab_runner", os.path.join(REPO, "scripts", "trm_colab_runner.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reference_step(param, grad, m, v, step, lr, beta1, beta2, wd):
    """adam_atan2_test._adam_atan2_reference_impl from adam-atan2 0.0.3 (fp64 here)."""
    param = param * (1 - lr * wd)
    m = torch.lerp(m, grad, 1 - beta1)
    v = v * beta2 + (1 - beta2) * grad * grad
    denom = v.sqrt() / math.sqrt(1 - beta2 ** step)
    param = param - (lr / (1 - beta1 ** step)) * torch.atan2(m, denom)
    return param, m, v


def test_adam_atan2_fallback_matches_reference():
    saved = sys.modules.pop("adam_atan2", None)
    try:
        runner = _load_runner()
        impl = runner._install_adam_atan2_fallback()
        if not impl.startswith("pure-torch"):
            pytest.skip("CUDA adam-atan2 extension is installed; fallback not exercised")
        AdamATan2 = sys.modules["adam_atan2"].AdamATan2

        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(64, 32, dtype=torch.float64) * 0.02)
        opt = AdamATan2([p], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
        ref_p, ref_m, ref_v = p.detach().clone(), torch.zeros_like(p), torch.zeros_like(p)
        for t in range(1, 51):
            g = torch.randn_like(p) * 1e-3
            p.grad = g.clone()
            opt.step()
            ref_p, ref_m, ref_v = _reference_step(ref_p, g, ref_m, ref_v, t, 1e-3, 0.9, 0.95, 0.1)
        assert torch.allclose(p.detach(), ref_p, atol=1e-12, rtol=0)
        st = opt.state[p]
        assert torch.allclose(st["exp_avg"], ref_m, atol=1e-15, rtol=0)
        assert torch.allclose(st["exp_avg_sq"], ref_v, atol=1e-15, rtol=0)
        assert float(st["step"]) == 50.0 and st["step"].dtype == torch.float32
    finally:
        sys.modules.pop("adam_atan2", None)
        if saved is not None:
            sys.modules["adam_atan2"] = saved


def test_config_fingerprint_ignores_volatile_fields():
    runner = _load_runner()
    a = runner._config_fingerprint({"lr": 1e-4, "run_name": "x", "checkpoint_path": "/a"})
    b = runner._config_fingerprint({"lr": 1e-4, "run_name": "y", "checkpoint_path": "/b"})
    c = runner._config_fingerprint({"lr": 2e-4, "run_name": "x", "checkpoint_path": "/a"})
    assert a == b and a != c
