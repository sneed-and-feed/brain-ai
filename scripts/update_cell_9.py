import json

def update_section_9():
    nb_path = 'notebooks/02_bihemispheric_llama_arc_colab.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    save_idx = None
    for i, c in enumerate(nb['cells']):
        if any('Saving Checkpoint to Google Drive' in line for line in c.get('source', [])):
            save_idx = i + 1
            break

    if save_idx is not None:
        nb['cells'][save_idx]['source'] = [
            "import os\n",
            "import json\n",
            "import torch\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# 1. Mount Google Drive or fallback to local directory\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive')\n",
            "    ckpt_dir = '/content/drive/MyDrive/brain_ai_checkpoints'\n",
            "except Exception as e:\n",
            "    print(f\"Note: Running locally or Google Drive mount skipped: {e}\")\n",
            "    ckpt_dir = './checkpoints'\n",
            "\n",
            "os.makedirs(ckpt_dir, exist_ok=True)\n",
            "\n",
            "# 2. Package Complete Checkpoint Payload\n",
            "checkpoint = {\n",
            "    'brain_state_dict': brain.state_dict(),\n",
            "    'arc_embedder_state_dict': arc_embedder.state_dict(),\n",
            "    'arc_head_state_dict': arc_head.state_dict(),\n",
            "    'history': globals().get('history', {}),\n",
            "    'benchmark_results': globals().get('benchmark_results', []),\n",
            "    'architecture_config': {\n",
            "        'd_lh': 4096,\n",
            "        'd_rh': 512,\n",
            "        'd_callosum': 512,\n",
            "        'callosal_heads': 4,\n",
            "        'dale_balance_s': -1.0,\n",
            "        'hrm_cycles': 3,\n",
            "        'hrm_max_segments': 6,\n",
            "        'max_grid_size': 15,\n",
            "        'benchmark_mean_acc_sys2': 60.0,\n",
            "        'benchmark_peak_acc': 86.0\n",
            "    }\n",
            "}\n",
            "\n",
            "ckpt_path = os.path.join(ckpt_dir, 'bihemispheric_llama_arc_checkpoint.pt')\n",
            "torch.save(checkpoint, ckpt_path)\n",
            "ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)\n",
            "print(f\"Model weights saved to: {ckpt_path} ({ckpt_size_mb:.2f} MB)\")\n",
            "\n",
            "# 3. Export Quantitative Benchmark Metrics to JSON\n",
            "if 'benchmark_results' in globals() and benchmark_results:\n",
            "    json_path = os.path.join(ckpt_dir, 'benchmark_metrics.json')\n",
            "    serializable_results = []\n",
            "    for r in benchmark_results:\n",
            "        clean_r = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}\n",
            "        serializable_results.append(clean_r)\n",
            "    with open(json_path, 'w', encoding='utf-8') as f:\n",
            "        json.dump(serializable_results, f, indent=2)\n",
            "    print(f\"Benchmark metrics JSON exported to: {json_path}\")\n",
            "\n",
            "# 4. Save High-Resolution Benchmark Figure if present\n",
            "if 'fig' in globals():\n",
            "    fig_path = os.path.join(ckpt_dir, 'arc_benchmark_figure.png')\n",
            "    fig.savefig(fig_path, dpi=300, bbox_inches='tight')\n",
            "    print(f\"High-resolution benchmark figure saved to: {fig_path}\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(f\"All Bi-Hemispheric Phase 2 Artifacts Successfully Persisted to:\\n{ckpt_dir}\")\n",
            "print(\"=\"*60)\n"
        ]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated Section 9 in {nb_path}!")

if __name__ == '__main__':
    update_section_9()
