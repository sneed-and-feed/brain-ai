#!/usr/bin/env python
"""
Resumable single-GPU launcher for the official TRM code (SamsungSAILMontreal/TinyRecursiveModels).

Why this exists
---------------
The official `pretrain.py` saves only model weights at evaluation time and has no resume path, which is
fragile under Colab session limits. This launcher reuses TRM's own functions (`create_dataloader`,
`init_train_state`, `train_batch`, `evaluate`, `create_evaluators`) unchanged and adds:

  * full-state resume (model incl. puzzle embeddings, optimizer moments, EMA shadow, step, data epoch)
    written atomically to a persistent directory (e.g. Google Drive) after every data iteration;
  * a wall-clock budget (`--max-hours`) that exits cleanly so the next session can resume;
  * decoupled evaluation frequency (`--eval-every-iters`) and an optional final evaluation on the full
    test split when interim evaluations use a subset (`data_paths_test`);
  * a JSON run manifest (commits, GPU, torch, overrides, optimizer implementation, resume history)
    and JSONL metric logs instead of Weights & Biases;
  * a throughput/memory probe mode (`--probe-steps N`) used by notebook 04 to size runs.

Deviations from the official training loop (all recorded in the manifest):
  * the ACT carry is re-initialised after a resume (one partial batch of halting state is lost);
  * ARC evaluators default to `aggregated_voting=False` (per-checkpoint voting, reproducible under resume);
    pass `--aggregated-voting` to restore the official behaviour of pooling votes across evaluations;
  * if the CUDA `adam-atan2` extension cannot be imported, a pure-PyTorch AdamATan2 with identical update
    rule and state layout is substituted (manifest field `optimizer_impl`).

Usage (from any directory):
  python scripts/trm_colab_runner.py --trm-dir /content/TinyRecursiveModels --out-dir /content/drive/.../run \
      [--max-hours 22] [--eval-every-iters 2] [--final-eval-full] [--probe-steps 0] -- <hydra overrides>
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
import torch

# Monkey-patch for adam-atan2 compatibility with PyTorch 2.4+
if not hasattr(torch.optim.Optimizer, '_cuda_graph_capture_health_check') and hasattr(torch.optim.Optimizer, '_accelerator_graph_capture_health_check'):
    torch.optim.Optimizer._cuda_graph_capture_health_check = torch.optim.Optimizer._accelerator_graph_capture_health_check


# ---------------------------------------------------------------------------
# Pure-PyTorch AdamATan2 fallback (update rule copied from adam-atan2 0.0.3 reference implementation)
# ---------------------------------------------------------------------------

def _install_adam_atan2_fallback() -> str:
    """Return 'cuda-extension' if adam_atan2 imports, else install a pure-torch module and return 'pure-torch'."""
    try:
        import adam_atan2  # noqa: F401
        from adam_atan2 import AdamATan2  # noqa: F401
        return "cuda-extension (adam-atan2)"
    except Exception:
        pass

    import torch
    from torch.optim.optimizer import Optimizer

    class AdamATan2(Optimizer):
        """
        param <- param * (1 - lr * wd)
        m <- lerp(m, g, 1 - beta1);  v <- beta2 * v + (1 - beta2) * g^2
        param <- param - (lr / bc1) * atan2(m, sqrt(v) / sqrt(bc2))
        State keys and dtypes match the CUDA extension, so checkpoints are interchangeable.
        """

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
    # Fields that may legitimately change between sessions are excluded.
    volatile = {"load_checkpoint", "checkpoint_path", "run_name", "project_name", "data_paths_test"}
    stable = {k: v for k, v in cfg_dict.items() if k not in volatile}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trm-dir", required=True, help="Path to a checkout of TinyRecursiveModels")
    ap.add_argument("--out-dir", required=True, help="Persistent run directory (resume state, logs, eval outputs)")
    ap.add_argument("--max-hours", type=float, default=None, help="Exit cleanly after this much wall-clock time")
    ap.add_argument("--eval-every-iters", type=int, default=1, help="Evaluate every N data iterations (and at the end)")
    ap.add_argument("--final-eval-full", action="store_true", help="After training, evaluate on the full test split of data_paths")
    ap.add_argument("--aggregated-voting", action="store_true", help="Official ARC behaviour: pool votes across evaluations")
    ap.add_argument("--keep-ckpts", type=int, default=2, help="Number of resume checkpoints to keep")
    ap.add_argument("--probe-steps", type=int, default=0, help="Run N training steps, report throughput/memory, then exit")
    ap.add_argument("--force-resume", action="store_true", help="Resume even if the config fingerprint differs")
    ap.add_argument("--leakage-guard", choices=["arc", "sudoku", "none"], default="none",
                    help="Run the built-dataset leakage check on data_paths[0] before training")
    ap.add_argument("--brain-ai-dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ap.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides for TRM's cfg_pretrain (after --)")
    args = ap.parse_args()
    overrides = [o for o in args.overrides if o != "--"]

    trm_dir = os.path.abspath(args.trm_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    resume_dir = os.path.join(out_dir, "resume")
    os.makedirs(resume_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "DONE.json")) and args.probe_steps == 0:
        print(f"[done] {out_dir} already completed (DONE.json present). Use a new --out-dir to start another run.")
        return

    optimizer_impl = _install_adam_atan2_fallback()

    os.chdir(trm_dir)
    sys.path.insert(0, trm_dir)
    os.environ.setdefault("WANDB_MODE", "disabled")

    import torch
    import torch.distributed as dist
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import pretrain as P  # official TRM training module

    with initialize_config_dir(config_dir=os.path.join(trm_dir, "config"), version_base=None):
        cfg = compose(config_name="cfg_pretrain", overrides=overrides)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = P.PretrainConfig(**cfg_dict)
    if config.project_name is None:
        config.project_name = os.path.basename(config.data_paths[0])
    if config.run_name is None:
        config.run_name = os.path.basename(out_dir)
    config.checkpoint_path = os.path.join(out_dir, "trm_outputs")  # evaluator outputs (submission.json) land here
    fingerprint = _config_fingerprint(config.model_dump())

    # Optional leakage guard on the built dataset
    if args.leakage_guard != "none" and args.probe_steps == 0:
        sys.path.insert(0, args.brain_ai_dir)
        from brain_ai.eval.leakage import assert_no_leakage, trm_built_dataset_overlap, trm_sudoku_overlap
        data_dir = config.data_paths[0] if os.path.isabs(config.data_paths[0]) else os.path.join(trm_dir, config.data_paths[0])
        rep = trm_built_dataset_overlap(data_dir) if args.leakage_guard == "arc" else trm_sudoku_overlap(data_dir)
        print(f"[leakage] {rep.summary()}")
        assert_no_leakage(rep, f"built dataset {data_dir}")

    # Single-process process group: TRM's ARC evaluator calls dist.gather_object.
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
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

    train_state = P.init_train_state(config, train_metadata, rank=0, world_size=1)
    n_params = sum(p.numel() for p in train_state.model.parameters())
    ema_helper = None
    if config.ema:
        ema_helper = P.EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # ------------------------------------------------------------------ resume
    iters_done = 0
    latest_ptr = os.path.join(resume_dir, "latest.json")
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"sessions": []}

    if os.path.exists(latest_ptr) and args.probe_steps == 0:
        ptr = json.load(open(latest_ptr))
        state = torch.load(os.path.join(resume_dir, ptr["file"]), map_location="cuda", weights_only=False)
        if state["fingerprint"] != fingerprint and not args.force_resume:
            raise RuntimeError(f"Config fingerprint changed ({state['fingerprint']} -> {fingerprint}); "
                               f"use a new --out-dir or pass --force-resume.")
        train_state.model.load_state_dict(state["model"])
        for opt, sd in zip(train_state.optimizers, state["optimizers"]):
            opt.load_state_dict(sd)
        if ema_helper is not None and state.get("ema") is not None:
            ema_helper.load_state_dict({k: v.to("cuda") for k, v in state["ema"].items()})
        train_state.step = int(state["step"])
        iters_done = int(state["iters_done"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        print(f"[resume] restored iteration {iters_done}/{total_iters}, step {train_state.step}")
    # Advance the dataset epoch counter so the data order continues where it stopped.
    train_loader.dataset._iters = iters_done * len(train_metadata.sets)

    session = {
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "resumed_from_iter": iters_done,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "optimizer_impl": optimizer_impl,
        "probe": args.probe_steps > 0,
    }
    manifest.update({
        "trm_commit": _git_commit(trm_dir),
        "brain_ai_commit": _git_commit(args.brain_ai_dir),
        "overrides": overrides,
        "config": config.model_dump(),
        "config_fingerprint": fingerprint,
        "n_params": n_params,
        "total_iters": total_iters,
        "total_steps_planned": train_state.total_steps,
        "aggregated_voting": bool(args.aggregated_voting),
        "deviations": [
            "ACT carry re-initialised after resume",
            f"ARC evaluator aggregated_voting={bool(args.aggregated_voting)} (official default: True)",
            f"optimizer: {optimizer_impl}",
        ],
    })
    if args.probe_steps == 0:
        manifest["sessions"].append(session)
        _atomic_json(manifest_path, manifest)

    metrics_log = os.path.join(out_dir, "train_metrics.jsonl")
    eval_log = os.path.join(out_dir, "eval_metrics.jsonl")

    # ------------------------------------------------------------------ probe mode
    if args.probe_steps > 0:
        train_state.model.train()
        torch.cuda.reset_peak_memory_stats()
        times, n = [], 0
        try:
            for _set, batch, gbs in train_loader:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                P.train_batch(config, train_state, batch, gbs, rank=0, world_size=1)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                n += 1
                if n >= args.probe_steps:
                    break
            warm = times[min(10, len(times) // 2):]  # exclude compilation warm-up
            sps = 1.0 / (sum(warm) / max(1, len(warm)))
            result = {"ok": True, "steps_per_sec": sps, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                      "global_batch_size": config.global_batch_size, "n_params": n_params,
                      "steps_per_epoch": train_state.total_steps / config.epochs, "optimizer_impl": optimizer_impl}
        except torch.cuda.OutOfMemoryError as e:
            result = {"ok": False, "error": "OOM", "detail": str(e)[:300], "global_batch_size": config.global_batch_size}
        _atomic_json(os.path.join(out_dir, f"probe_bs{config.global_batch_size}.json"), result)
        print("[probe]", json.dumps(_to_jsonable(result)))
        dist.destroy_process_group()
        return

    # ------------------------------------------------------------------ training loop
    def run_eval(tag: str, loader, meta, evs):
        if loader is None:
            return None
        if ema_helper is not None:
            eval_model = ema_helper.ema_copy(train_state.model)
        else:
            eval_model = train_state.model
        eval_state = P.TrainState(model=eval_model, optimizers=[], optimizer_lrs=[], carry=None,
                                  step=train_state.step, total_steps=train_state.total_steps)
        eval_state.model.eval()
        t0 = time.time()
        metrics = P.evaluate(config, eval_state, loader, meta, evs, rank=0, world_size=1, cpu_group=cpu_group)
        rec = {"tag": tag, "iter": iters_done, "step": train_state.step, "eval_seconds": round(time.time() - t0, 1),
               "metrics": metrics}
        _append_jsonl(eval_log, rec)
        print(f"[eval:{tag}] step {train_state.step}: {json.dumps(_to_jsonable(metrics))}")
        del eval_model
        torch.cuda.empty_cache()
        return metrics

    def save_resume():
        fname = f"iter_{iters_done:05d}.pt"
        tmp = os.path.join(resume_dir, fname + ".tmp")
        torch.save({
            "model": train_state.model.state_dict(),
            "optimizers": [o.state_dict() for o in train_state.optimizers],
            "ema": ema_helper.state_dict() if ema_helper is not None else None,
            "step": train_state.step,
            "iters_done": iters_done,
            "fingerprint": fingerprint,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }, tmp)
        os.replace(tmp, os.path.join(resume_dir, fname))
        _atomic_json(latest_ptr, {"file": fname, "iters_done": iters_done, "step": train_state.step,
                                  "saved": _dt.datetime.now().isoformat(timespec="seconds")})
        for old in sorted(glob.glob(os.path.join(resume_dir, "iter_*.pt")))[:-args.keep_ckpts]:
            os.remove(old)

    t_session = time.time()
    log_every = 50
    t_last_log = time.time()
    buf = []
    finished = iters_done >= total_iters
    print(f"[train] Starting training: {total_iters} iterations total, ~{train_state.total_steps} steps planned.", flush=True)
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

        if iters_done % args.eval_every_iters == 0 or iters_done == total_iters:
            run_eval("interim" if iters_done < total_iters else "final", eval_loader, eval_metadata, evaluators)
        save_resume()

        if args.max_hours is not None and (time.time() - t_session) / 3600.0 > args.max_hours and iters_done < total_iters:
            print(f"[budget] {args.max_hours} h reached; state saved at iteration {iters_done}. Re-run to resume.")
            break
    finished = iters_done >= total_iters

    if finished:
        if ema_helper is not None:
            final_model = ema_helper.ema_copy(train_state.model)
            torch.save(final_model.state_dict(), os.path.join(out_dir, "final_ema_weights.pt"))
            del final_model
        else:
            torch.save(train_state.model.state_dict(), os.path.join(out_dir, "final_weights.pt"))
        if args.final_eval_full and config.data_paths_test:
            full_cfg = config.model_copy(update={"data_paths_test": []})
            loader, meta, evs = make_eval(full_cfg)
            run_eval("final_full_test", loader, meta, evs)
        _atomic_json(os.path.join(out_dir, "DONE.json"), {"finished": _dt.datetime.now().isoformat(timespec="seconds"),
                                                          "step": train_state.step, "iters": iters_done})
    session["ended"] = _dt.datetime.now().isoformat(timespec="seconds")
    session["ended_at_iter"] = iters_done
    manifest["sessions"][-1] = session
    _atomic_json(manifest_path, manifest)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
