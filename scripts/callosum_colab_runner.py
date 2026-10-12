#!/usr/bin/env python
"""
Resumable single-GPU launcher for the Bi-Hemispheric Architecture (Phase P2 / RQ1).

Couples a frozen Left Hemisphere (LH) LLM representation to the TRM recursive spatial reasoner
via a Dale's-law-constrained cross-attention Corpus Callosum:
  1. Offline cached LH latents replace the learned puzzle-ID embedding table.
  2. Callosal Conditioning options:
     - 'dale_inhibitory' (s = -1.0, DaleLinear, DaleDifferentialCrossAttention)
     - 'dale_excitatory' (s = +1.0, DaleLinear, DaleDifferentialCrossAttention)
     - 'unconstrained' (s = +1.0, standard Linear, MultiheadAttention)
  3. Streamlined evaluation:
     - Fast interim checks during training (loss, token accuracy, fast Pass@2) taking ~1-2 min.
     - Full multi-attempt evaluation at the final checkpoint (step 5,010) for 100% academic parity against B1.
  4. Full-state atomic Google Drive checkpointing & resume.
  5. Built-in leakage guard and JSON run manifests.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Monkey-patch for adam-atan2 compatibility with PyTorch 2.4+
if not hasattr(torch.optim.Optimizer, '_cuda_graph_capture_health_check') and hasattr(torch.optim.Optimizer, '_accelerator_graph_capture_health_check'):
    torch.optim.Optimizer._cuda_graph_capture_health_check = torch.optim.Optimizer._accelerator_graph_capture_health_check

if sys.version_info >= (3, 13) or "COLAB_GPU" in os.environ:
    os.environ.setdefault("DISABLE_COMPILE", "1")


# ---------------------------------------------------------------------------
# Pure-PyTorch AdamATan2 fallback
# ---------------------------------------------------------------------------

def _install_adam_atan2_fallback() -> str:
    try:
        import adam_atan2  # noqa: F401
        from adam_atan2 import AdamATan2  # noqa: F401
        return "cuda-extension (adam-atan2)"
    except Exception:
        pass

    from torch.optim.optimizer import Optimizer

    class AdamATan2(Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2):
            if lr < 0.0 or not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0 or weight_decay < 0.0:
                raise ValueError("Invalid AdamATan2 hyperparameters")
            super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                lr, wd = group["lr"], group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["step"].add_(1)
                    t = float(state["step"].item())
                    work_dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
                    param = p.to(work_dtype)
                    grad = p.grad.to(work_dtype)
                    m = state["exp_avg"].to(work_dtype)
                    v = state["exp_avg_sq"].to(work_dtype)
                    param.mul_(1.0 - lr * wd)
                    m.lerp_(grad, 1.0 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = v.sqrt() / math.sqrt(1.0 - beta2 ** t)
                    param.add_(torch.atan2(m, denom), alpha=-(lr / (1.0 - beta1 ** t)))
                    p.copy_(param)
                    state["exp_avg"].copy_(m)
                    state["exp_avg_sq"].copy_(v)

    module = types.ModuleType("adam_atan2")
    module.AdamATan2 = AdamATan2
    sys.modules["adam_atan2"] = module
    return "pure-torch fallback (identical update rule; slower)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _to_jsonable(x):
    try:
        import numpy as np
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "item") and callable(x.item):
        try:
            return x.item()
        except Exception:
            return str(x)
    return x


def _atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path: str, obj) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(_to_jsonable(obj)) + "\n")


def _config_fingerprint(cfg_dict: dict) -> str:
    volatile = {"load_checkpoint", "checkpoint_path", "run_name", "project_name", "data_paths_test"}
    stable = {k: v for k, v in cfg_dict.items() if k not in volatile}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _clean_keys(d):
    if not isinstance(d, dict):
        return d
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in d.items()}


def _normalize_keys(sd: dict, target_keys) -> dict:
    has_orig = any(k.startswith("_orig_mod.") for k in target_keys)
    res = {}
    for k, v in sd.items():
        clean = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        new_k = ("_orig_mod." + clean) if has_orig else clean
        res[new_k] = v
    return res


# ---------------------------------------------------------------------------
# Main Launcher
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trm-dir", required=True, help="Path to TinyRecursiveModels checkout")
    ap.add_argument("--out-dir", required=True, help="Persistent run directory (checkpoints, logs, outputs)")
    ap.add_argument("--lh-cache-path", default=None, help="Path to cached_lh_latents.pt")
    ap.add_argument("--constraint-type", choices=["dale_inhibitory", "dale_excitatory", "unconstrained"],
                    default="dale_inhibitory", help="Callosal conditioning constraint type")
    ap.add_argument("--coupling-sign", type=float, default=-1.0, help="Transcallosal cross-talk sign (-1.0 or +1.0)")
    ap.add_argument("--callosal-heads", type=int, default=4, help="Attention heads for Callosal Conditioning")
    ap.add_argument("--d-lh", type=int, default=4096, help="LH representation latent dimension")
    ap.add_argument("--d-callosum", type=int, default=512, help="Callosal manifold dimension")
    ap.add_argument("--p-excitatory", type=float, default=0.8, help="Excitatory unit fraction for DaleLinear")
    ap.add_argument("--interim-eval-fast", action="store_true",
                    help="Use lightweight interim checks during training; run full evaluator only at final step")
    ap.add_argument("--run-tta", action="store_true",
                    help="Run System 2 Test-Time Latent Adaptation (continuous latent relaxation delta_z) on evaluation tasks")
    ap.add_argument("--tta-steps", type=int, default=30, help="Number of Adam steps for latent relaxation delta_z")
    ap.add_argument("--tta-lr", type=float, default=0.01, help="Learning rate for latent relaxation delta_z")
    ap.add_argument("--tta-anchor", type=float, default=1.0, help="Proximal anchor quadratic regularization weight lambda_anchor")
    ap.add_argument("--eval-only", action="store_true", help="Skip training and run final evaluation only")
    ap.add_argument("--weights-path", default=None, help="Explicit path to model weights checkpoint to load in eval-only mode")
    ap.add_argument("--eval-tasks-path", default=None, help="Optional path to evaluation challenges JSON")
    ap.add_argument("--eval-solutions-path", default=None, help="Optional path to evaluation solutions JSON")
    ap.add_argument("--compare-s1-s2", action="store_true", default=True,
                    help="Generate and compare System 1 Zero-Shot vs System 2 TTA submissions")
    ap.add_argument("--skip-trm-eval-batches", action="store_true",
                    help="Bypass TRM's slow 61-batch recurrent evaluation loop and use direct test predictions")
    ap.add_argument("--max-hours", type=float, default=None, help="Exit cleanly after this much wall-clock time")
    ap.add_argument("--eval-every-iters", type=int, default=1, help="Evaluate every N data iterations")
    ap.add_argument("--aggregated-voting", action="store_true", help="Pool votes across evaluations")
    ap.add_argument("--keep-ckpts", type=int, default=2, help="Number of resume checkpoints to keep")
    ap.add_argument("--probe-steps", type=int, default=0, help="Run N probe steps, report throughput/mem, exit")
    ap.add_argument("--force-resume", action="store_true", help="Resume even if config fingerprint differs")
    ap.add_argument("--leakage-guard", choices=["arc", "sudoku", "none"], default="none",
                    help="Run built-dataset leakage check before training")
    ap.add_argument("--brain-ai-dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ap.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides for TRM cfg_pretrain (after --)")
    args = ap.parse_args()
    overrides = [o for o in args.overrides if o != "--"]

    trm_dir = os.path.abspath(args.trm_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    resume_dir = os.path.join(out_dir, "resume")
    os.makedirs(resume_dir, exist_ok=True)

    if os.path.exists(os.path.join(out_dir, "DONE.json")) and args.probe_steps == 0 and not args.eval_only:
        print(f"[done] {out_dir} already completed (DONE.json present).")
        return

    optimizer_impl = _install_adam_atan2_fallback()

    os.chdir(trm_dir)
    sys.path.insert(0, trm_dir)
    sys.path.insert(0, args.brain_ai_dir)
    os.environ.setdefault("WANDB_MODE", "disabled")

    import torch.distributed as dist
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import pretrain as P
    from brain_ai.models.callosum import (
        CallosalConditioning,
        CallosalPuzzleEmbedding,
        System2LatentAdaptor,
        find_callosal_embedding,
        wire_callosal_embedding
    )
    from brain_ai.models.lh_cache import load_lh_cache
    from brain_ai.eval.arc_scorer import load_kaggle_split, score_submission
    from brain_ai.eval.stats import wilson_interval, mcnemar_exact

    with initialize_config_dir(config_dir=os.path.join(trm_dir, "config"), version_base=None):
        cfg = compose(config_name="cfg_pretrain", overrides=overrides)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = P.PretrainConfig(**cfg_dict)
    if args.eval_only and config.global_batch_size > 128:
        print(f"[eval_only] Capping global_batch_size from {config.global_batch_size} to 128 for evaluation memory bounds.", flush=True)
        config.global_batch_size = 128
    if config.project_name is None:
        config.project_name = os.path.basename(config.data_paths[0])
    if config.run_name is None:
        config.run_name = os.path.basename(out_dir)
    config.checkpoint_path = os.path.join(out_dir, "callosum_outputs")
    fingerprint = _config_fingerprint(config.model_dump())

    # Leakage guard
    if args.leakage_guard != "none" and args.probe_steps == 0:
        from brain_ai.eval.leakage import assert_no_leakage, trm_built_dataset_overlap, trm_sudoku_overlap
        data_dir = config.data_paths[0] if os.path.isabs(config.data_paths[0]) else os.path.join(trm_dir, config.data_paths[0])
        rep = trm_built_dataset_overlap(data_dir) if args.leakage_guard == "arc" else trm_sudoku_overlap(data_dir)
        print(f"[leakage] {rep.summary()}")
        assert_no_leakage(rep, f"built dataset {data_dir}")

    # Process group initialization
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=0, world_size=1)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    cpu_group = dist.new_group(backend="gloo")
    torch.random.manual_seed(config.seed)

    iter_epochs = config.eval_interval if config.eval_interval is not None else config.epochs
    if config.epochs % iter_epochs != 0:
        raise ValueError("eval_interval must divide epochs (TRM constraint)")
    total_iters = config.epochs // iter_epochs

    train_loader, train_metadata = P.create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=iter_epochs,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)

    def make_eval(cfg_for_eval):
        try:
            loader, meta = P.create_dataloader(cfg_for_eval, "test", test_set_mode=True, epochs_per_iter=1,
                                               global_batch_size=cfg_for_eval.global_batch_size, rank=0, world_size=1)
        except FileNotFoundError:
            return None, None, []
        evs = P.create_evaluators(cfg_for_eval, meta) if cfg_for_eval.evaluators else []
        for ev in evs:
            if hasattr(ev, "aggregated_voting"):
                ev.aggregated_voting = bool(args.aggregated_voting)
        return loader, meta, evs

    eval_loader, eval_metadata, evaluators = make_eval(config)

    # Initialize standard TRM base model
    train_state = P.init_train_state(config, train_metadata, rank=0, world_size=1)

    # ------------------------------------------------------------------
    # Wire Callosal Conditioning & LH Latents
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_puzzles = getattr(train_metadata, "num_puzzles", 400)

    # Load or generate cached LH latents
    if args.lh_cache_path and os.path.exists(args.lh_cache_path):
        print(f"[callosum] Loading LH cached latents from: {args.lh_cache_path}")
        cached_latents, lh_meta = load_lh_cache(args.lh_cache_path, device=device)
        d_lh = cached_latents.shape[-1]
    else:
        print(f"[callosum] LH cache path not specified or not found. Initializing mock LH latents ({num_puzzles}, 32, {args.d_lh}).")
        g = torch.Generator().manual_seed(42)
        cached_latents = torch.randn(num_puzzles, 32, args.d_lh, generator=g, device=device)
        cached_latents = F.layer_norm(cached_latents, (args.d_lh,))
        d_lh = args.d_lh

    puzzle_emb_len = getattr(config.arch, "puzzle_emb_len", 16)
    d_reasoner = getattr(config.arch, "hidden_size", 512)

    callosum_cond = CallosalConditioning(
        d_lh=d_lh,
        d_reasoner=d_reasoner,
        d_callosum=args.d_callosum,
        num_query_tokens=puzzle_emb_len,
        n_heads=args.callosal_heads,
        p_excitatory=args.p_excitatory,
        cross_talk_sign=args.coupling_sign,
        constraint_type=args.constraint_type
    ).to(device)

    puzzle_to_base = lh_meta.get("puzzle_to_base_id") if isinstance(lh_meta, dict) else None
    callosal_emb = CallosalPuzzleEmbedding(
        cached_lh_latents=cached_latents,
        callosal_conditioning=callosum_cond,
        num_query_tokens=puzzle_emb_len,
        d_reasoner=d_reasoner,
        flatten_output=True,
        puzzle_to_base_id=puzzle_to_base
    ).to(device)

    # Universally wire the Callosal bridge across all model wrappers and inner reasoning layers
    wire_callosal_embedding(train_state.model, callosal_emb)
    train_state.model.to(device)

    # Re-initialize optimizers to include Callosal Conditioning parameters
    callosum_param_set = set(callosum_cond.parameters())
    reasoner_params = [p for p in train_state.model.parameters() if p not in callosum_param_set]
    callosum_params = list(callosum_cond.parameters())

    import adam_atan2
    train_state.optimizers = [
        adam_atan2.AdamATan2(reasoner_params, lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay),
        adam_atan2.AdamATan2(callosum_params, lr=config.puzzle_emb_lr, betas=(config.beta1, config.beta2), weight_decay=config.puzzle_emb_weight_decay)
    ]

    n_params = sum(p.numel() for p in train_state.model.parameters())
    n_callosum_params = sum(p.numel() for p in callosum_cond.parameters())
    print(f"[callosum] Bi-Hemispheric Reasoner wired! Total params: {n_params:,} (Callosum: {n_callosum_params:,})")

    ema_helper = None
    if config.ema:
        ema_helper = P.EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # ------------------------------------------------------------------ Resume
    iters_done = 0
    latest_ptr = os.path.join(resume_dir, "latest.json")
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"sessions": []}

    if os.path.exists(latest_ptr) and args.probe_steps == 0:
        ptr = json.load(open(latest_ptr))
        state = torch.load(os.path.join(resume_dir, ptr["file"]), map_location=device, weights_only=False)
        target_keys = train_state.model.state_dict().keys()
        train_state.model.load_state_dict(_normalize_keys(state["model"], target_keys), strict=False)
        for opt, sd in zip(train_state.optimizers, state["optimizers"]):
            opt.load_state_dict(sd)
        if ema_helper is not None and state.get("ema") is not None:
            norm_ema = _normalize_keys(state["ema"], ema_helper.shadow.keys())
            ema_helper.load_state_dict({k: v.to(device) for k, v in norm_ema.items()})
        train_state.step = int(state["step"])
        iters_done = int(state["iters_done"])
        if "torch_rng" in state:
            try:
                torch.set_rng_state(state["torch_rng"].cpu().to(torch.uint8))
            except Exception as e:
                print(f"[resume] warning: could not restore torch_rng: {e}")
        if "cuda_rng" in state and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state(state["cuda_rng"].cpu().to(torch.uint8))
            except Exception as e:
                print(f"[resume] warning: could not restore cuda_rng: {e}")
        print(f"[resume] restored iteration {iters_done}/{total_iters}, step {train_state.step}")

    train_loader.dataset._iters = iters_done * len(train_metadata.sets)

    session = {
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "resumed_from_iter": iters_done,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else 0.0,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "none",
        "optimizer_impl": optimizer_impl,
        "probe": args.probe_steps > 0,
        "constraint_type": args.constraint_type,
        "coupling_sign": args.coupling_sign,
        "n_callosum_params": n_callosum_params
    }
    manifest.update({
        "trm_commit": _git_commit(trm_dir),
        "brain_ai_commit": _git_commit(args.brain_ai_dir),
        "overrides": overrides,
        "config": config.model_dump(),
        "config_fingerprint": fingerprint,
        "n_params": n_params,
        "n_callosum_params": n_callosum_params,
        "total_iters": total_iters,
        "total_steps_planned": train_state.total_steps,
        "aggregated_voting": bool(args.aggregated_voting),
        "interim_eval_fast": bool(args.interim_eval_fast),
        "constraint_type": args.constraint_type,
        "coupling_sign": args.coupling_sign
    })
    if args.probe_steps == 0:
        manifest["sessions"].append(session)
        _atomic_json(manifest_path, manifest)

    metrics_log = os.path.join(out_dir, "train_metrics.jsonl")
    eval_log = os.path.join(out_dir, "eval_metrics.jsonl")

    # ------------------------------------------------------------------ Probe mode
    if args.probe_steps > 0:
        train_state.model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        times, n = [], 0
        try:
            for _set, batch, gbs in train_loader:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                P.train_batch(config, train_state, batch, gbs, rank=0, world_size=1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                n += 1
                if n >= args.probe_steps:
                    break
            warm = times[min(10, len(times) // 2):]
            sps = 1.0 / (sum(warm) / max(1, len(warm)))
            peak_mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            result = {
                "ok": True, "steps_per_sec": sps, "peak_mem_gb": peak_mem,
                "global_batch_size": config.global_batch_size, "n_params": n_params,
                "n_callosum_params": n_callosum_params,
                "steps_per_epoch": train_state.total_steps / config.epochs, "optimizer_impl": optimizer_impl
            }
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                result = {"ok": False, "error": "OOM", "detail": str(e)[:300], "global_batch_size": config.global_batch_size}
            else:
                raise
        _atomic_json(os.path.join(out_dir, f"probe_bs{config.global_batch_size}.json"), result)
        print("[probe]", json.dumps(_to_jsonable(result)))
        dist.destroy_process_group()
        return

    # ------------------------------------------------------------------ Streamlined Evaluation Logic
    def run_eval(tag: str, is_final: bool = False):
        if eval_loader is None:
            return None
        if ema_helper is not None and not args.eval_only:
            eval_model = ema_helper.ema_copy(train_state.model)
            wire_callosal_embedding(eval_model, callosal_emb)
        else:
            eval_model = train_state.model
            wire_callosal_embedding(eval_model, callosal_emb)
        eval_state = P.TrainState(model=eval_model, optimizers=[], optimizer_lrs=[], carry=None,
                                  step=train_state.step, total_steps=train_state.total_steps)
        eval_state.model.eval()
        t0 = time.time()

        # Streamlined evaluation protocol:
        # If fast interim evaluation is enabled and this is NOT the final iteration:
        # evaluate without evaluators (loss + token accuracy only, taking <1 min instead of 1.8h).
        # At final step (is_final=True), run full evaluators [1, 2, 5, 10, 100] for 100% academic parity against B1.
        if args.interim_eval_fast and not is_final:
            active_evaluators = []
        else:
            active_evaluators = evaluators

        # If final evaluation and run_tta is requested, run System 2 continuous latent adaptation
        if is_final and args.run_tta:
            print("[tta] Initializing System 2 Test-Time Latent Adaptation (continuous latent relaxation delta_z)...", flush=True)
            adaptor = System2LatentAdaptor(
                n_steps=args.tta_steps,
                lr=args.tta_lr,
                lambda_anchor=args.tta_anchor,
                device=device
            )
            # Find evaluation challenges
            challenges_path = args.eval_tasks_path
            solutions_path = args.eval_solutions_path
            eval_challenges = None
            eval_solutions = None
            if challenges_path and os.path.exists(challenges_path):
                with open(challenges_path) as f:
                    eval_challenges = json.load(f)
                if solutions_path and os.path.exists(solutions_path):
                    with open(solutions_path) as f:
                        eval_solutions = json.load(f)
            else:
                search_candidates = [
                    os.path.join(trm_dir, "kaggle", "combined", "arc-agi_evaluation_challenges.json"),
                    os.path.join(trm_dir, "kaggle", "combined", "arc-agi", "evaluation"),
                    os.path.join(trm_dir, "data", "arc-agi-1-aug", "test_puzzles.json"),
                    os.path.join(trm_dir, config.data_paths[0], "test_puzzles.json") if config.data_paths else "",
                    os.path.join(args.brain_ai_dir, "data", "arc", "evaluation")
                ]
                for cand in search_candidates:
                    if cand and os.path.exists(cand):
                        if os.path.isdir(cand):
                            eval_challenges = {}
                            eval_solutions = {}
                            for fname in sorted(os.listdir(cand)):
                                if fname.endswith(".json"):
                                    tid = fname[:-5]
                                    with open(os.path.join(cand, fname)) as jf:
                                        tdata = json.load(jf)
                                    eval_challenges[tid] = {"train": tdata.get("train", []), "test": [{"input": t["input"]} for t in tdata.get("test", [])]}
                                    if all("output" in t for t in tdata.get("test", [])):
                                        eval_solutions[tid] = [t["output"] for t in tdata.get("test", [])]
                        else:
                            with open(cand) as f:
                                raw_puzzles = json.load(f)
                            eval_challenges = {tid: {"train": p.get("train", []), "test": [{"input": t["input"]} for t in p.get("test", [])]}
                                               for tid, p in raw_puzzles.items()}
                            eval_solutions = None
                            sol_cand = cand.replace("_challenges.json", "_solutions.json")
                            if os.path.exists(sol_cand):
                                with open(sol_cand) as sf:
                                    eval_solutions = json.load(sf)
                            elif all("output" in t for p in raw_puzzles.values() for t in p.get("test", [])):
                                eval_solutions = {tid: [t["output"] for t in p.get("test", [])] for tid, p in raw_puzzles.items()}
                        break

            if eval_challenges:
                identifiers_path = os.path.join(trm_dir, config.data_paths[0], "identifiers.json") if config.data_paths else ""
                identifiers_map = None
                if os.path.exists(identifiers_path):
                    with open(identifiers_path) as f:
                        identifiers_map = json.load(f)

                print(f"[tta] Running latent relaxation for {len(eval_challenges)} evaluation tasks ({args.tta_steps} Adam steps, lambda_anchor={args.tta_anchor})...", flush=True)
                tta_start = time.time()
                tta_diags = {}
                accepted_count = 0

                id_to_idx = {}
                if isinstance(identifiers_map, list):
                    id_to_idx = {name: i for i, name in enumerate(identifiers_map)}
                elif isinstance(identifiers_map, dict):
                    id_to_idx = {k: int(v) for k, v in identifiers_map.items()}

                sub_s1 = {}
                sub_s2 = {}

                for task_idx, (tid, task_data) in enumerate(eval_challenges.items()):
                    pid = id_to_idx.get(tid, 0)
                    if tid not in id_to_idx and id_to_idx:
                        print(f"[tta] Warning: Task '{tid}' not found in identifiers map; defaulting pid to 0.")

                    # Predict System 1 Zero-Shot (unadapted delta_z = 0)
                    sub_s1[tid] = []
                    for t in task_data.get("test", []):
                        pred_s1 = adaptor.predict_test_pair(eval_model, t["input"], puzzle_id=pid, delta_z=None)
                        sub_s1[tid].append({"attempt_1": pred_s1, "attempt_2": pred_s1})

                    # Perform System 2 Test-Time Latent Adaptation (delta_z continuous relaxation)
                    delta_z, diag = adaptor.adapt_task(
                        model=eval_model,
                        task_data=task_data,
                        puzzle_id=pid,
                        n_steps=args.tta_steps,
                        lr=args.tta_lr,
                        lambda_anchor=args.tta_anchor
                    )
                    tta_diags[tid] = diag

                    # Predict System 2 Adapted (uses delta_z if accepted, else falls back to System 1)
                    sub_s2[tid] = []
                    for t_idx, t in enumerate(task_data.get("test", [])):
                        if diag["accepted"]:
                            pred_s2 = adaptor.predict_test_pair(eval_model, t["input"], puzzle_id=pid, delta_z=delta_z)
                        else:
                            pred_s2 = sub_s1[tid][t_idx]["attempt_1"]
                        sub_s2[tid].append({"attempt_1": pred_s2, "attempt_2": sub_s1[tid][t_idx]["attempt_1"]})

                    if diag["accepted"]:
                        accepted_count += 1
                        for m in (eval_model, train_state.model):
                            call_emb = find_callosal_embedding(m)
                            if call_emb is not None and hasattr(call_emb, "set_puzzle_delta_z"):
                                call_emb.set_puzzle_delta_z(pid, delta_z)
                                if getattr(call_emb, "puzzle_to_base_id", None) is not None and pid < len(call_emb.puzzle_to_base_id):
                                    base_pid = int(call_emb.puzzle_to_base_id[pid].item())
                                    call_emb.set_puzzle_delta_z(base_pid, delta_z)

                    if (task_idx + 1) % 50 == 0 or (task_idx + 1) == len(eval_challenges):
                        print(f"  [tta] Adapted {task_idx + 1}/{len(eval_challenges)} tasks | {accepted_count} accepted ({100*accepted_count/(task_idx+1):.1f}%) | elapsed {time.time() - tta_start:.1f}s", flush=True)

                tta_duration = time.time() - tta_start
                print(f"[tta] Completed adaptation in {tta_duration:.1f}s: {accepted_count}/{len(eval_challenges)} tasks accepted ({100*accepted_count/max(1, len(eval_challenges)):.1f}%).", flush=True)

                # Save dual submission artifacts
                s1_sub_path = os.path.join(out_dir, "submission_s1_zero_shot.json")
                s2_sub_path = os.path.join(out_dir, "submission_s2_adapted.json")
                legacy_sub_path = os.path.join(out_dir, "system2_tta_submission.json")
                _atomic_json(s1_sub_path, sub_s1)
                _atomic_json(s2_sub_path, sub_s2)
                _atomic_json(legacy_sub_path, sub_s2)
                print(f"[tta] Saved System 1 Zero-Shot submission to: {s1_sub_path}", flush=True)
                print(f"[tta] Saved System 2 Adapted submission to: {s2_sub_path}", flush=True)

                _atomic_json(os.path.join(out_dir, "tta_adaptation_summary.json"), {
                    "n_tasks": len(eval_challenges),
                    "accepted_tasks": accepted_count,
                    "acceptance_rate": accepted_count / max(1, len(eval_challenges)),
                    "tta_duration_seconds": tta_duration,
                    "tta_steps": args.tta_steps,
                    "tta_lr": args.tta_lr,
                    "lambda_anchor": args.tta_anchor,
                    "per_task_diagnostics": tta_diags
                })

                if eval_solutions:
                    print("\n" + "="*70, flush=True)
                    print("=== HEAD-TO-HEAD: SYSTEM 1 ZERO-SHOT vs SYSTEM 2 TTA ===", flush=True)
                    print("="*70, flush=True)
                    rep_s1 = score_submission(sub_s1, eval_solutions)
                    rep_s2 = score_submission(sub_s2, eval_solutions)
                    lo_s1, hi_s1 = wilson_interval(rep_s1.tasks_fully_solved, rep_s1.n_tasks)
                    lo_s2, hi_s2 = wilson_interval(rep_s2.tasks_fully_solved, rep_s2.n_tasks)
                    print(f"System 1 (Zero-Shot): {rep_s1.tasks_fully_solved}/{rep_s1.n_tasks} ({100*rep_s1.tasks_fully_solved/rep_s1.n_tasks:.2f}%) [{100*lo_s1:.2f}%, {100*hi_s1:.2f}%] 95% Wilson CI", flush=True)
                    print(f"System 2 (Adapted):   {rep_s2.tasks_fully_solved}/{rep_s2.n_tasks} ({100*rep_s2.tasks_fully_solved/rep_s2.n_tasks:.2f}%) [{100*lo_s2:.2f}%, {100*hi_s2:.2f}%] 95% Wilson CI", flush=True)
                    tids = sorted(eval_solutions.keys())
                    s1_corr = [rep_s1.per_task.get(t, 0.0) == 1.0 for t in tids]
                    s2_corr = [rep_s2.per_task.get(t, 0.0) == 1.0 for t in tids]
                    mc_res = mcnemar_exact(s2_corr, s1_corr)
                    print(f"McNemar Test (S2 vs S1): S2 only={mc_res['only_a']}, S1 only={mc_res['only_b']}, two-sided p={mc_res['p_value']:.4f}", flush=True)
                    print("="*70 + "\n", flush=True)

                    _atomic_json(os.path.join(out_dir, "system2_vs_system1_summary.json"), {
                        "n_tasks": len(eval_challenges),
                        "accepted_count": accepted_count,
                        "acceptance_rate": accepted_count / max(1, len(eval_challenges)),
                        "system1": {
                            "tasks_fully_solved": rep_s1.tasks_fully_solved,
                            "percentage": 100 * rep_s1.tasks_fully_solved / rep_s1.n_tasks,
                            "wilson_ci": [lo_s1, hi_s1]
                        },
                        "system2": {
                            "tasks_fully_solved": rep_s2.tasks_fully_solved,
                            "percentage": 100 * rep_s2.tasks_fully_solved / rep_s2.n_tasks,
                            "wilson_ci": [lo_s2, hi_s2]
                        },
                        "mcnemar_s2_vs_s1": mc_res
                    })

                if args.skip_trm_eval_batches:
                    print("[eval] --skip-trm-eval-batches specified: bypassing slow TRM multi-attempt dataloader batches.", flush=True)
                    metrics = {
                        "system1_exact": rep_s1.tasks_fully_solved if 'rep_s1' in locals() else 0,
                        "system2_exact": rep_s2.tasks_fully_solved if 'rep_s2' in locals() else 0
                    }
                    eval_duration = round(time.time() - t0, 1)
                    rec = {
                        "tag": tag,
                        "iter": iters_done,
                        "step": train_state.step,
                        "eval_seconds": eval_duration,
                        "is_final": is_final,
                        "metrics": metrics
                    }
                    _append_jsonl(eval_log, rec)
                    del eval_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return metrics
            else:
                raise FileNotFoundError(f"[tta] ERROR: --run-tta was requested, but evaluation challenges could not be located in any candidate path: {search_candidates}")

        metrics = P.evaluate(config, eval_state, eval_loader, eval_metadata, active_evaluators, rank=0, world_size=1, cpu_group=cpu_group)
        eval_duration = round(time.time() - t0, 1)

        if is_final and args.run_tta:
            eval_sub_paths = glob.glob(os.path.join(config.checkpoint_path, "evaluator_ARC_step_*", "submission.json"))
            if not eval_sub_paths:
                eval_sub_paths = glob.glob(os.path.join(out_dir, "**", "submission.json"), recursive=True)
            if eval_sub_paths:
                latest_sub = sorted(eval_sub_paths, key=os.path.getmtime)[-1]
                dest_sub = os.path.join(out_dir, "system2_tta_submission.json")
                import shutil
                shutil.copy2(latest_sub, dest_sub)
                print(f"[tta] Copied System 2 TTA submission to: {dest_sub}", flush=True)

        rec = {
            "tag": tag,
            "iter": iters_done,
            "step": train_state.step,
            "eval_seconds": eval_duration,
            "is_final": is_final,
            "metrics": metrics
        }
        _append_jsonl(eval_log, rec)
        print(f"[eval:{tag}] step {train_state.step} ({eval_duration}s): {json.dumps(_to_jsonable(metrics))}", flush=True)

        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    def save_resume():
        fname = f"iter_{iters_done:05d}.pt"
        tmp = os.path.join(resume_dir, fname + ".tmp")
        torch.save({
            "model": _clean_keys(train_state.model.state_dict()),
            "optimizers": [o.state_dict() for o in train_state.optimizers],
            "ema": _clean_keys(ema_helper.state_dict()) if ema_helper is not None else None,
            "step": train_state.step,
            "iters_done": iters_done,
            "fingerprint": fingerprint,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }, tmp)
        os.replace(tmp, os.path.join(resume_dir, fname))
        _atomic_json(latest_ptr, {"file": fname, "iters_done": iters_done, "step": train_state.step,
                                  "saved": _dt.datetime.now().isoformat(timespec="seconds")})
        for old in sorted(glob.glob(os.path.join(resume_dir, "iter_*.pt")))[:-args.keep_ckpts]:
            os.remove(old)

    # ------------------------------------------------------------------ Eval-Only Mode
    if args.eval_only:
        print("[eval_only] Running evaluation-only mode...", flush=True)
        loaded = False
        candidates = []
        if args.weights_path:
            candidates.append(args.weights_path)
        candidates.extend([
            os.path.join(out_dir, "final_ema_weights.pt"),
            os.path.join(out_dir, "final_weights.pt"),
        ])
        resumes = sorted(glob.glob(os.path.join(resume_dir, "iter_*.pt")))
        if resumes:
            candidates.append(resumes[-1])

        for path in candidates:
            if os.path.exists(path):
                print(f"[eval_only] Loading weights from {path}")
                raw_sd = torch.load(path, map_location=device, weights_only=False)
                sd = raw_sd.get("model", raw_sd) if isinstance(raw_sd, dict) else raw_sd
                train_state.model.load_state_dict(_normalize_keys(sd, train_state.model.state_dict().keys()), strict=False)
                wire_callosal_embedding(train_state.model, callosal_emb)
                loaded = True
                break

        done_path = os.path.join(out_dir, "DONE.json")
        if os.path.exists(done_path):
            try:
                with open(done_path) as f:
                    done_info = json.load(f)
                train_state.step = int(done_info.get("step", train_state.step))
                print(f"[eval_only] Restored step {train_state.step} from DONE.json")
            except Exception:
                pass

        if not loaded:
            print("[eval_only] WARNING: No checkpoint weights found to load! Evaluating current model state.", flush=True)

        run_eval("eval_only", is_final=True)
        dist.destroy_process_group()
        return

    # ------------------------------------------------------------------ Training Loop
    t_session = time.time()
    log_every = 50
    t_last_log = time.time()
    buf = []
    print(f"[train] Starting Bi-Hemispheric Callosal training: {total_iters} iterations total, ~{train_state.total_steps} steps planned.", flush=True)

    while iters_done < total_iters:
        train_state.model.train()
        t_iter = time.time()
        for _set, batch, gbs in train_loader:
            m = P.train_batch(config, train_state, batch, gbs, rank=0, world_size=1)
            if ema_helper is not None:
                ema_helper.update(train_state.model)
            if m is not None:
                buf.append(m)
                if len(buf) >= log_every:
                    avg = {k: float(sum(float(b[k]) for b in buf) / len(buf)) for k in buf[0]}
                    avg["step"] = train_state.step
                    _append_jsonl(metrics_log, avg)
                    elapsed = max(time.time() - t_last_log, 1e-4)
                    sps = len(buf) / elapsed
                    loss_keys = [k for k in avg if "loss" in k]
                    loss_val = avg[loss_keys[0]] if loss_keys else 0.0
                    print(f"  [step {train_state.step:5d}/{train_state.total_steps}] loss={loss_val:.4f} | {sps:.1f} steps/s", flush=True)
                    t_last_log = time.time()
                    buf = []
        iters_done += 1
        print(f"[train] iteration {iters_done}/{total_iters} done in {time.time() - t_iter:.0f}s (step {train_state.step})", flush=True)

        is_last = (iters_done == total_iters)
        if iters_done % args.eval_every_iters == 0 or is_last:
            run_eval("interim" if not is_last else "final", is_final=is_last)
        save_resume()

        if args.max_hours is not None and (time.time() - t_session) / 3600.0 > args.max_hours and iters_done < total_iters:
            print(f"[budget] {args.max_hours} h reached; state saved at iteration {iters_done}. Re-run to resume.")
            break

    finished = iters_done >= total_iters

    if finished:
        if ema_helper is not None:
            final_model = ema_helper.ema_copy(train_state.model)
            torch.save(_clean_keys(final_model.state_dict()), os.path.join(out_dir, "final_ema_weights.pt"))
            del final_model
        else:
            torch.save(_clean_keys(train_state.model.state_dict()), os.path.join(out_dir, "final_weights.pt"))
        _atomic_json(os.path.join(out_dir, "DONE.json"), {"finished": _dt.datetime.now().isoformat(timespec="seconds"),
                                                          "step": train_state.step, "iters": iters_done})

    session["ended"] = _dt.datetime.now().isoformat(timespec="seconds")
    session["ended_at_iter"] = iters_done
    manifest["sessions"][-1] = session
    _atomic_json(manifest_path, manifest)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
