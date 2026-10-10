import json
import os

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "A100",
                "provenance": []
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    def add_md(source):
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(source):
        lines = [line + "\n" for line in source.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    add_md(r"""# Notebook 04: Project Integrity & B1 Baseline (TRM)
    
This notebook establishes the rigorous baseline floor for the Bi-Hemispheric AI project (P0 gate). 

## Protocol
1. **Integrity Checks (Part A):** We verify the exact-match scorer, compute no-model statistical floors using the true ARC scorer (which does *not* provide the output shape to the model), execute a formal leakage audit on the Kaggle-format data, and formally record the nullity of the previous Phase 2 progress (which relied on an oracle-shape per-pixel metric and leaked evaluation labels).
2. **Sudoku Baseline (Part B):** We train the TRM Sudoku MLP. Because training takes ~18h on an L40S, this defaults to a pilot/partial run.
3. **ARC B1 Baseline (Part C):** We run the Tiny Recursive Model (TRM) on ARC-AGI-1.
   **Crucial Disclosure:** TRM's data encoding injects a learned "puzzle-ID embedding" token. This token learns a dedicated embedding for every puzzle, *including evaluation tasks*, trained directly on the evaluation task's demonstrations. This violates strict few-shot isolation. We treat TRM (Baseline B1) as a reference point for *test-time-adapted* (TTA) encoding, against which our frozen-LLM baseline (B3) will be compared.
""")

    add_code(r"""import os

# "smoke" = rapid CI check (minutes)
# "pilot" = reduced epochs/augmentations to test full workflow (10-20 A100-hours)
# "full" = exact README figures (days)
RUN_MODE = "pilot" 

DRIVE_ROOT = "/content/drive/MyDrive/brain_ai_P0"
TRM_COMMIT = "c01103738605ba39d1430519b1ee0c62f4c707f8"
SEED = 42

# Mode-specific parameters
if RUN_MODE == "smoke":
    SUDOKU_SUBSAMPLE = 100
    SUDOKU_AUG = 10
    SUDOKU_EPOCHS = 100
    SUDOKU_TEST_SUBSET = 100
    
    ARC_AUG = 8
    
elif RUN_MODE == "pilot":
    SUDOKU_SUBSAMPLE = 1000
    SUDOKU_AUG = 1000
    SUDOKU_EPOCHS = 10000  # 20% of the 50000 target
    SUDOKU_TEST_SUBSET = 20000
    
    ARC_AUG = 300  # instead of 1000
    
else:  # full
    SUDOKU_SUBSAMPLE = 1000
    SUDOKU_AUG = 1000
    SUDOKU_EPOCHS = 50000
    SUDOKU_TEST_SUBSET = 100000
    
    ARC_AUG = 1000

os.environ["RUN_MODE"] = RUN_MODE
os.environ["DRIVE_ROOT"] = DRIVE_ROOT
os.environ["TRM_COMMIT"] = TRM_COMMIT
os.environ["SUDOKU_SUBSAMPLE"] = str(SUDOKU_SUBSAMPLE)
os.environ["SUDOKU_AUG"] = str(SUDOKU_AUG)
os.environ["SUDOKU_EPOCHS"] = str(SUDOKU_EPOCHS)
os.environ["ARC_AUG"] = str(ARC_AUG)
""")

    add_md("## Setup & Installation")

    add_code(r"""from google.colab import drive
from google.colab import userdata
import subprocess
import os

drive.mount('/content/drive')

!nvidia-smi

# Authenticate with Hugging Face using the token stored in Colab Secrets
hf_token = userdata.get('HF_TOKEN')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    !huggingface-cli login --token $HF_TOKEN
else:
    print("HF_TOKEN not found in Colab secrets. Skipping Hugging Face login.")

# Clone brain-ai
if not os.path.exists("/content/brain-ai"):
    !git clone https://github.com/sneed-and-feed/brain-ai.git /content/brain-ai
else:
    !cd /content/brain-ai && git pull

# Clone TRM and pin to the exact commit
if not os.path.exists("/content/TinyRecursiveModels"):
    !git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git /content/TinyRecursiveModels
    !cd /content/TinyRecursiveModels && git checkout $TRM_COMMIT
else:
    # Ensure we are on the correct commit even if the cell is rerun
    !cd /content/TinyRecursiveModels && git fetch && git checkout $TRM_COMMIT

# Install adam-atan2 (CUDA extension) and other TRM requirements
!pip install -q einops coolname pydantic argdantic wandb omegaconf hydra-core huggingface_hub pytest setuptools_scm ninja wheel packaging

# Attempt adam-atan2 install from source
!CUDA_HOME=/usr/local/cuda pip install --no-cache-dir --no-build-isolation adam-atan2 || echo "CUDA adam-atan2 failed, runner will use pure-torch fallback"
""")

    add_md("## Part A: Integrity & Audit")
    
    add_code(r"""import sys
sys.path.insert(0, "/content/brain-ai")

import json
import zipfile
import time
from brain_ai.eval.arc_scorer import load_kaggle_split, score_floors
from brain_ai.eval.leakage import raw_arc_overlap
from brain_ai.eval.stats import wilson_interval
from brain_ai.eval.legacy import rescore_legacy_progress

# 1. Run eval tests
!uv pip install pytest || pip install pytest
!cd /content/brain-ai && python -m pytest tests/test_eval.py tests/test_trm_runner.py -q

# 2. Extract dataset paths
TRM_KAGGLE_DIR = "/content/TinyRecursiveModels/kaggle/combined/arc-agi"
splits = {s: load_kaggle_split(TRM_KAGGLE_DIR, s) for s in ["training", "evaluation", "concept", "training2", "evaluation2"]}

print(f"\n--- Leakage Audit ---")
# ARC-1 Eval vs TRM training splits
r1 = raw_arc_overlap(*splits["evaluation"], {"training": splits["training"], "concept": splits["concept"]})
print(f"ARC-1 eval vs training+concept: {r1.summary()}")
if r1.leaked_test_pairs:
    print("   Note: Task 070dd51e/40853293 is a known ARC benchmark-intrinsic duplicate.")

# Positive Control
rpc = raw_arc_overlap(*splits["evaluation"], {"training2": splits["training2"]})
print(f"POSITIVE CONTROL (ARC-1 eval vs training2): {rpc.summary()}")
assert len(rpc.leaked_test_pairs) > 300, "Positive control failed!"

r2 = raw_arc_overlap(*splits["evaluation2"], {"training2": splits["training2"], "concept": splits["concept"]})
print(f"ARC-2 eval vs training2+concept: {r2.summary()}")

print(f"\n--- Statistical Floors ---")
t0 = time.time()
for s in ["evaluation", "evaluation2"]:
    rep = score_floors(*splits[s])
    for name, r in rep.items():
        lo, hi = wilson_interval(r.tasks_fully_solved, r.n_tasks)
        print(f"floor {s:12s} {name:30s} {r.summary()}  wilson[{100*lo:.1f},{100*hi:.1f}]")
print(f"Floors computed in {time.time()-t0:.1f}s")

print(f"\n--- Legacy Phase 2 Nullity Record ---")
legacy_path = "/content/brain-ai/checkpoints/phase2/eval400_progress.json"
if os.path.exists(legacy_path):
    rescore = rescore_legacy_progress(legacy_path)
    print("Official exact-match re-score of previous pipeline:")
    print(json.dumps(rescore, indent=2))
else:
    print(f"Legacy record {legacy_path} not found.")
""")

    add_md("## Part B: Sudoku MLP Baseline")

    add_code(r"""%%bash
# Build Sudoku dataset
cd /content/TinyRecursiveModels

if [ ! -d "data/sudoku-extreme-built" ]; then
    echo "Building Sudoku dataset..."
    python dataset/build_sudoku_dataset.py \
        --output-dir data/sudoku-extreme-built \
        --subsample-size ${SUDOKU_SUBSAMPLE} \
        --num-aug ${SUDOKU_AUG}
fi
""")

    add_code(r"""import numpy as np
import os
import shutil

# Subset the Sudoku test set for faster interim evaluations
src = "/content/TinyRecursiveModels/data/sudoku-extreme-built/test"
dst = "/content/TinyRecursiveModels/data/sudoku-test-subset/test"
os.makedirs(dst, exist_ok=True)

if not os.path.exists(os.path.join(dst, "dataset.json")):
    print("Slicing Sudoku test set...")
    shutil.copy(os.path.join(src, "dataset.json"), os.path.join(dst, "dataset.json"))
    for f in os.listdir(src):
        if f.endswith(".npy"):
            arr = np.load(os.path.join(src, f), mmap_mode="r")
            subset = arr[:SUDOKU_TEST_SUBSET+1] if "indices" in f else arr[:SUDOKU_TEST_SUBSET]
            np.save(os.path.join(dst, f), subset)
    shutil.copy("/content/TinyRecursiveModels/data/sudoku-extreme-built/identifiers.json", 
                "/content/TinyRecursiveModels/data/sudoku-test-subset/identifiers.json")
""")

    add_code(r"""%%bash
# Run Sudoku
cd /content/brain-ai
export PYTHONPATH=/content/TinyRecursiveModels:$PYTHONPATH
export DISABLE_COMPILE=1
export PYTHONUNBUFFERED=1

python -u scripts/trm_colab_runner.py \
    --trm-dir /content/TinyRecursiveModels \
    --out-dir ${DRIVE_ROOT}/sudoku_mlp \
    --max-hours 11.5 \
    --eval-every-iters 5000 \
    --final-eval-full \
    --leakage-guard sudoku \
    -- \
    arch=trm evaluators="[]" \
    epochs=${SUDOKU_EPOCHS} \
    eval_interval=1000 \
    lr=1e-4 puzzle_emb_lr=1e-4 \
    weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.mlp_t=True arch.pos_encodings=none \
    arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
    ema=True \
    data_paths="[data/sudoku-extreme-built]" \
    data_paths_test="[data/sudoku-test-subset]"
""")

    add_md("## Part C: ARC-AGI-1 Baseline B1")

    add_code(r"""%%bash
cd /content/TinyRecursiveModels

if [ ! -d "data/arc-agi-1-aug" ]; then
    echo "Building ARC dataset..."
    python -m dataset.build_arc_dataset \
        --input-file-prefix kaggle/combined/arc-agi \
        --output-dir data/arc-agi-1-aug \
        --subsets training evaluation concept \
        --test-set-name evaluation \
        --num-aug ${ARC_AUG}
fi
""")

    add_code(r"""import subprocess
import json
import os

print("Probing batch sizes to find memory bounds...")
bs_to_try = [768, 512, 384, 256, 128]
optimal_bs = None
steps_per_epoch = None

for bs in bs_to_try:
    print(f"Trying global_batch_size={bs}...")
    cmd = [
        "python", "/content/brain-ai/scripts/trm_colab_runner.py",
        "--trm-dir", "/content/TinyRecursiveModels",
        "--out-dir", "/content/probe_run",
        "--probe-steps", "15",
        "--",
        "arch=trm", f"data_paths=[data/arc-agi-1-aug]",
        "arch.L_layers=2", "arch.H_cycles=3", "arch.L_cycles=4",
        "ema=True", f"global_batch_size={bs}"
    ]
    subprocess.run(cmd, capture_output=True)
    probe_file = f"/content/probe_run/probe_bs{bs}.json"
    if os.path.exists(probe_file):
        with open(probe_file) as f:
            res = json.load(f)
        if res.get("ok"):
            print(f"  Success: {res['steps_per_sec']:.2f} steps/s, Peak memory {res['peak_mem_gb']:.1f} GB")
            optimal_bs = bs
            steps_per_epoch = res['steps_per_epoch']
            break
        else:
            print(f"  OOM: {res.get('detail')}")

if optimal_bs is None:
    raise RuntimeError("Could not find a batch size that fits in memory.")

# Compute epochs based on the probe.
# We want evaluations roughly every 30-45 minutes.
eval_interval_epochs = max(1, int((30 * 60 * res['steps_per_sec']) / steps_per_epoch))
# For a pilot, do 4 evals total. For full, standard TRM is 100000 epochs.
target_epochs = eval_interval_epochs * 4 if os.environ["RUN_MODE"] == "pilot" else 100000

print(f"Selected batch size: {optimal_bs}")
print(f"Eval interval: {eval_interval_epochs} epochs. Total epochs: {target_epochs}.")

# Write config parameters out for the next cell
with open("/content/arc_run_config.sh", "w") as f:
    f.write(f"export ARC_BS={optimal_bs}\n")
    f.write(f"export ARC_EPOCHS={target_epochs}\n")
    f.write(f"export ARC_EVAL_INT={eval_interval_epochs}\n")
""")

    add_code(r"""%%bash
source /content/arc_run_config.sh
cd /content/brain-ai
export PYTHONPATH=/content/TinyRecursiveModels:$PYTHONPATH
export DISABLE_COMPILE=1
export PYTHONUNBUFFERED=1

python -u scripts/trm_colab_runner.py \
    --trm-dir /content/TinyRecursiveModels \
    --out-dir ${DRIVE_ROOT}/arc_trm_b1 \
    --max-hours 11.5 \
    --eval-every-iters 1 \
    --leakage-guard arc \
    -- \
    arch=trm \
    data_paths="[data/arc-agi-1-aug]" \
    arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
    ema=True \
    global_batch_size=${ARC_BS} \
    epochs=${ARC_EPOCHS} \
    eval_interval=${ARC_EVAL_INT}
""")

    add_code(r"""import glob
import json
import os
import sys

sys.path.insert(0, "/content/brain-ai")
from brain_ai.eval.arc_scorer import load_kaggle_split, score_submission
from brain_ai.eval.stats import wilson_interval

out_dir = os.path.join(os.environ["DRIVE_ROOT"], "arc_trm_b1")
done_file = os.path.join(out_dir, "DONE.json")

if os.path.exists(done_file):
    print("Run completed successfully. Rescoring submission...")
    sub_paths = glob.glob(os.path.join(out_dir, "trm_outputs", "evaluator_ARC_step_*", "submission.json"))
    
    if sub_paths:
        latest_sub = sorted(sub_paths)[-1]
        print(f"Evaluating {latest_sub}")
        
        with open(latest_sub) as f:
            submission = json.load(f)
            
        _, solutions = load_kaggle_split("/content/TinyRecursiveModels/kaggle/combined/arc-agi", "evaluation")
        
        report = score_submission(submission, solutions)
        lo, hi = wilson_interval(report.tasks_fully_solved, report.n_tasks)
        
        print("\n=== FINAL B1 RESULTS (INDEPENDENT SCORER) ===")
        print(report.summary())
        print(f"Exact match (fully solved tasks): {report.tasks_fully_solved}/{report.n_tasks} [{100*lo:.1f}%, {100*hi:.1f}%] 95% CI")
        
        with open(os.path.join(out_dir, "b1_eval_summary.json"), "w") as f:
            json.dump({
                "score": report.score,
                "n_tasks": report.n_tasks,
                "tasks_fully_solved": report.tasks_fully_solved,
                "wilson_interval": [lo, hi],
                "submission_file": os.path.basename(os.path.dirname(latest_sub))
            }, f, indent=2)
    else:
        print("No submission.json found.")
else:
    print("Run is not yet finished (or DONE.json missing).")
""")

    add_md(r"""## Post-Run Checklist
Once this notebook completes successfully in `pilot` mode, the environment and data flow are verified.
The resulting metrics form the strict `B1` baseline for ARC-AGI-1.
""")

    with open("notebooks/04_integrity_and_trm_baseline_colab.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

if __name__ == "__main__":
    create_notebook()
    print("Generated notebooks/04_integrity_and_trm_baseline_colab.ipynb")
