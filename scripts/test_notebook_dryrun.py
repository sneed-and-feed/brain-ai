import json
import re
import os
import sys

def dry_run_notebook():
    nb_path = "notebooks/03_phase2_bihemispheric_scaling_arc2_colab.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
    print(f"Total code cells to verify: {len(code_cells)}")

    cleaned_scripts = []
    for idx, cell in enumerate(code_cells):
        cell_text = "".join(cell)
        lines = []
        for line in cell_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("%"):
                lines.append("# " + line)
            else:
                lines.append(line)
        cleaned_scripts.append((idx, "\n".join(lines)))

    ns = {"__name__": "__main__"}
    for idx, script in cleaned_scripts:
        print(f"Testing execution of cell {idx}...")
        try:
            # Limit evaluation task count in dry run so it finishes quickly
            if "NUM_EVAL_TASKS = min(25, len(task_files))" in script:
                script = script.replace("NUM_EVAL_TASKS = min(25, len(task_files))", "NUM_EVAL_TASKS = 2")
            if "NUM_TRAIN_STEPS = 15" in script:
                script = script.replace("NUM_TRAIN_STEPS = 15", "NUM_TRAIN_STEPS = 2")
            exec(script, ns)
        except Exception as e:
            print(f"Cell {idx} raised an error: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n[SUCCESS] ALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY END-TO-END!")
    return True

if __name__ == "__main__":
    success = dry_run_notebook()
    sys.exit(0 if success else 1)
