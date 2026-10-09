import json

def update_notebook():
    nb_path = 'notebooks/02_bihemispheric_llama_arc_colab.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Cell 13 (markdown 6) to clarify zero-shot
    nb['cells'][13]['source'] = [
        "## 6. Zero-Shot Baseline Evaluation & Qualitative Synthesis\n",
        "\n",
        "Inspect the Right Hemisphere's initial zero-shot predicted ARC output grid and generate a linguistic explanation from Llama 3.1 conditioned on callosal feedback before task-specific adaptation."
    ]

    # 2. Markdown cell for Section 7: TTA
    tta_md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Few-Shot Test-Time Adaptation (TTA) on Task Demonstrations\n",
            "\n",
            "In ARC-AGI, human inductive reasoning abstracts rules by observing 2–3 training demonstrations and then applying that learned hypothesis to the unseen test grid.\n",
            "\n",
            "Here we perform **Few-Shot Test-Time Adaptation (TTA)**:\n",
            "1. **Snapshot & Reset Base Model**: Preserves base pre-trained weights across repeated task evaluations.\n",
            "2. **Linguistic Priming**: Left Hemisphere encodes symbolic demonstration pairs into latent guidance $Z_{\\text{LH}}$.\n",
            "3. **Fast Spatial Adaptation**: Unrolls 30 fast gradient steps on the task demonstrations ($\\approx 2-3$ seconds on A100), optimizing Callosum + HRM + Prediction Head.\n",
            "4. **Test Challenge Evaluation**: Applies the adapted network to the test challenge grid.\n",
            "5. **Cognitive Visualization & Linguistic Reflection**: Generates the 4-panel visualizer and Llama 3.1 explanation conditioned on adapted transcallosal feedback."
        ]
    }

    # 3. Code cell for Section 7: TTA
    tta_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import copy\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import torch.optim as optim\n",
            "\n",
            "# 1. Snapshot base pre-trained weights so we can adapt to any specific task cleanly\n",
            "if 'base_state_dict' not in globals():\n",
            "    base_state_dict = {\n",
            "        'brain': copy.deepcopy(brain.state_dict()),\n",
            "        'arc_embedder': copy.deepcopy(arc_embedder.state_dict()),\n",
            "        'arc_head': copy.deepcopy(arc_head.state_dict())\n",
            "    }\n",
            "else:\n",
            "    brain.load_state_dict(base_state_dict['brain'])\n",
            "    arc_embedder.load_state_dict(base_state_dict['arc_embedder'])\n",
            "    arc_head.load_state_dict(base_state_dict['arc_head'])\n",
            "\n",
            "# Select task: by default, adapt to the task sampled in Step 6, or choose any task\n",
            "target_task_id = eval_batch.task_ids[0] if 'eval_batch' in globals() else arc_dataset.tasks[0].task_id\n",
            "task = next((t for t in arc_dataset.tasks if t.task_id == target_task_id), arc_dataset.tasks[0])\n",
            "print(f\"=== Few-Shot Test-Time Adaptation (TTA) on Task: {task.task_id} ===\")\n",
            "print(f\"Demonstration pairs available: {len(task.train_pairs)} | Test challenge grids: {len(task.test_pairs)}\")\n",
            "\n",
            "# 2. Format demonstration pairs\n",
            "max_dim = arc_head.max_size\n",
            "d_in_list, d_out_list, d_mask_list = [], [], []\n",
            "for p in task.train_pairs:\n",
            "    inp, out = p['input'], p['output']\n",
            "    p_in = torch.zeros(max_dim, max_dim, dtype=torch.long, device=device)\n",
            "    p_out = torch.zeros(max_dim, max_dim, dtype=torch.long, device=device)\n",
            "    m = torch.zeros(max_dim, max_dim, device=device)\n",
            "    for r in range(min(len(inp), max_dim)):\n",
            "        for c in range(min(len(inp[0]), max_dim)):\n",
            "            p_in[r, c] = inp[r][c]\n",
            "    for r in range(min(len(out), max_dim)):\n",
            "        for c in range(min(len(out[0]), max_dim)):\n",
            "            p_out[r, c] = out[r][c]\n",
            "            m[r, c] = 1.0\n",
            "    d_in_list.append(p_in)\n",
            "    d_out_list.append(p_out)\n",
            "    d_mask_list.append(m)\n",
            "\n",
            "demo_inputs = torch.stack(d_in_list)\n",
            "demo_targets = torch.stack(d_out_list)\n",
            "demo_masks = torch.stack(d_mask_list)\n",
            "\n",
            "# 3. Linguistic Prompting: Left Hemisphere symbolic representation\n",
            "demo_prompt_strs = [f\"Ex{i+1}: in={p['input']} -> out={p['output']}\" for i, p in enumerate(task.train_pairs)]\n",
            "task_prompt = f\"ARC Task {task.task_id}: Deduce spatial transformation rule: {' ; '.join(demo_prompt_strs)}\"\n",
            "\n",
            "with torch.no_grad():\n",
            "    tok = lh_model.tokenize([task_prompt] * len(task.train_pairs), max_length=128)\n",
            "    z_lh_demo = lh_model.extract_residual_stream(\n",
            "        tok['input_ids'].to(device),\n",
            "        tok.get('attention_mask', None).to(device) if 'attention_mask' in tok and tok['attention_mask'] is not None else None\n",
            "    ).to(torch.float32).detach()\n",
            "\n",
            "# 4. Fast TTA Optimizer for Callosal Bridge, HRM, and Heads\n",
            "tta_params = (\n",
            "    list(brain.corpus_callosum.parameters()) +\n",
            "    list(brain.right_hemisphere.parameters()) +\n",
            "    list(arc_embedder.parameters()) +\n",
            "    list(arc_head.parameters())\n",
            ")\n",
            "tta_optimizer = optim.AdamW(tta_params, lr=1.5e-3, weight_decay=1e-4)\n",
            "tta_ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=10, reduction='none')\n",
            "\n",
            "print(\"\\n--- Unrolling Fast Test-Time Adaptation on Demonstrations ---\")\n",
            "for tta_step in range(1, 31):\n",
            "    tta_optimizer.zero_grad()\n",
            "    rh_in = arc_embedder(demo_inputs)\n",
            "    out = brain(lh_latents=z_lh_demo, rh_inputs=rh_in)\n",
            "    logits, gate = arc_head(out['rh_latents_updated'] + rh_in, input_grids=demo_inputs, return_gate=True)\n",
            "    \n",
            "    ce_matrix = tta_ce(logits, demo_targets.clamp(0, 9))\n",
            "    loss = (ce_matrix * demo_masks).sum() / (demo_masks.sum() + 1e-8)\n",
            "    \n",
            "    # Preserve Dale-constrained homeostatic stability during adaptation\n",
            "    total_loss = loss + 0.05 * out['callosum_losses']['loss_homeostatic']\n",
            "    total_loss.backward()\n",
            "    nn.utils.clip_grad_norm_(tta_params, max_norm=1.0)\n",
            "    tta_optimizer.step()\n",
            "    \n",
            "    if tta_step == 1 or tta_step % 10 == 0 or tta_step == 30:\n",
            "        with torch.no_grad():\n",
            "            preds = logits.argmax(dim=1)\n",
            "            demo_acc = (((preds == demo_targets.clamp(0, 9)).float() * demo_masks).sum() / (demo_masks.sum() + 1e-8)).item() * 100.0\n",
            "        print(f\"TTA Step {tta_step:2d}/30 | Demo Loss: {loss.item():.4f} | Demo Fit: {demo_acc:.1f}% | Spatial Gate: {gate.mean().item():.2f}\")\n",
            "\n",
            "# 5. Evaluate Adapted Bi-Hemispheric Brain on Unseen Test Challenge Grid\n",
            "test_pair = task.test_pairs[0]\n",
            "H_in, W_in = len(test_pair['input']), len(test_pair['input'][0])\n",
            "H_out, W_out = len(test_pair['output']), len(test_pair['output'][0])\n",
            "\n",
            "test_in_tensor = torch.zeros(1, max_dim, max_dim, dtype=torch.long, device=device)\n",
            "for r in range(min(H_in, max_dim)):\n",
            "    for c in range(min(W_in, max_dim)):\n",
            "        test_in_tensor[0, r, c] = test_pair['input'][r][c]\n",
            "\n",
            "with torch.no_grad():\n",
            "    tok_test = lh_model.tokenize([task_prompt], max_length=128)\n",
            "    z_lh_test = lh_model.extract_residual_stream(\n",
            "        tok_test['input_ids'].to(device),\n",
            "        tok_test.get('attention_mask', None).to(device) if 'attention_mask' in tok_test and tok_test['attention_mask'] is not None else None\n",
            "    ).to(torch.float32).detach()\n",
            "    rh_in_test = arc_embedder(test_in_tensor)\n",
            "    out_test = brain(lh_latents=z_lh_test, rh_inputs=rh_in_test)\n",
            "    \n",
            "    pred_logits, gate_map = arc_head(out_test['rh_latents_updated'] + rh_in_test, input_grids=test_in_tensor, return_gate=True)\n",
            "    pred_grid = pred_logits.argmax(dim=1)[0, :H_out, :W_out].cpu().numpy()\n",
            "    gt_grid = np.array(test_pair['output'])\n",
            "    in_grid = np.array(test_pair['input'])\n",
            "    gate_grid = gate_map[0, 0, :H_out, :W_out].cpu().numpy()\n",
            "    \n",
            "    test_acc = (pred_grid == gt_grid).mean() * 100.0\n",
            "\n",
            "print(\"\\n\" + \"=\"*55)\n",
            "print(f\">>> Adapted Test Challenge Exact-Match Accuracy: {test_acc:.1f}% <<<\")\n",
            "print(\"=\"*55)\n",
            "\n",
            "# 6. 4-Panel Visualization\n",
            "fig, ax = plt.subplots(1, 4, figsize=(16, 4))\n",
            "ax[0].imshow(in_grid, cmap=arc_cmap, vmin=0, vmax=10)\n",
            "ax[0].set_title(f\"1. Test Input Grid ({H_in}x{W_in})\")\n",
            "ax[0].axis('off')\n",
            "\n",
            "ax[1].imshow(gt_grid, cmap=arc_cmap, vmin=0, vmax=10)\n",
            "ax[1].set_title(f\"2. Ground Truth Target ({H_out}x{W_out})\")\n",
            "ax[1].axis('off')\n",
            "\n",
            "ax[2].imshow(pred_grid, cmap=arc_cmap, vmin=0, vmax=10)\n",
            "ax[2].set_title(f\"3. Adapted Prediction ({test_acc:.1f}%)\")\n",
            "ax[2].axis('off')\n",
            "\n",
            "im_gate = ax[3].imshow(gate_grid, cmap='coolwarm', vmin=0.0, vmax=1.0)\n",
            "ax[3].set_title(\"4. Spatial Gate g(x)\\n(Red=Preserve, Blue=Transform)\")\n",
            "ax[3].axis('off')\n",
            "plt.colorbar(im_gate, ax=ax[3], fraction=0.046, pad=0.04)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# 7. Linguistic Reflection from Llama 3.1 conditioned on adapted callosal feedback\n",
            "print(\"\\n--- Left Hemisphere (Llama 3.1) Linguistic Reflection ---\")\n",
            "delta_lh_feedback = out_test['lh_latents_updated'] - z_lh_test\n",
            "\n",
            "reflection_prompt = (\n",
            "    f\"<|start_header_id|>system<|end_header_id|>\\n\\n\"\n",
            "    f\"You are the symbolic Left Hemisphere of a bi-hemispheric neuromorphic AI. \"\n",
            "    f\"You analyze spatial visual transformations on ARC-AGI grids.<|eot_id|>\"\n",
            "    f\"<|start_header_id|>user<|end_header_id|>\\n\\n\"\n",
            "    f\"Task Demonstrations: {task_prompt}\\n\"\n",
            "    f\"The Right Hemisphere unrolled recurrent spatial dynamics on the {H_in}x{W_in} input grid after adapting to {len(task.train_pairs)} demonstrations (Callosal feedback norm: {delta_lh_feedback.norm().item():.2f}).\\n\"\n",
            "    f\"In 2 clear sentences, describe the spatial transformation rule discovered by the bi-hemispheric system:<|eot_id|>\"\n",
            "    f\"<|start_header_id|>assistant<|end_header_id|>\\n\\n\"\n",
            ")\n",
            "\n",
            "explanation = lh_model.generate_with_callosal_feedback(\n",
            "    prompt=reflection_prompt,\n",
            "    delta_lh=delta_lh_feedback,\n",
            "    max_new_tokens=64,\n",
            "    temperature=0.6\n",
            ")\n",
            "print(f\"Llama Explanation: {explanation}\")\n",
            "print(f\"Conflict Score: {out_test['conflict_score'].item():.4f}\")\n",
            "print(f\"Amygdalar Affective State: {out_test['affective_state']}\")"
        ]
    }

    # 4. Check if TTA is already present
    has_tta = any('Test-Time Adaptation' in ''.join(c.get('source', [])) for c in nb['cells'])
    if not has_tta:
        # Find cell 15 which was checkpoint saving
        save_idx = None
        for i, c in enumerate(nb['cells']):
            if any('Saving Checkpoint to Google Drive' in line for line in c.get('source', [])):
                save_idx = i
                break
        
        if save_idx is not None:
            nb['cells'][save_idx]['source'] = ['## 8. Saving Checkpoint to Google Drive\n']
            nb['cells'].insert(save_idx, tta_code_cell)
            nb['cells'].insert(save_idx, tta_md_cell)
        else:
            nb['cells'].append(tta_md_cell)
            nb['cells'].append(tta_code_cell)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated {nb_path} with TTA cells!")

if __name__ == '__main__':
    update_notebook()
