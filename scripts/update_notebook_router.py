import json
import copy

def update_notebook_benchmark():
    nb_path = 'notebooks/02_bihemispheric_llama_arc_colab.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    bench_cell = None
    bench_idx = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code' and any('Scaled Multi-Condition Benchmark' in line for line in c.get('source', [])):
            bench_cell = c
            bench_idx = i
            break

    if bench_cell is None:
        raise ValueError("Could not find benchmark cell in notebook!")

    src = "".join(bench_cell['source'])

    # Let's inspect what needs replacement
    old_target = """    # -------------------------------------------------------------------
    # Condition 2: Bi-Hemispheric System 1 (Fast Feedforward Reflex)
    # -------------------------------------------------------------------
    brain.load_state_dict(base_state_dict['brain'])
    arc_embedder.load_state_dict(base_state_dict['arc_embedder'])
    arc_head.load_state_dict(base_state_dict['arc_head'])
    
    t0 = time.time()
    with torch.no_grad():
        tok = lh_model.tokenize([task_prompt], max_length=128)
        z_lh_test = lh_model.extract_residual_stream(
            tok['input_ids'].to(device),
            tok.get('attention_mask', None).to(device) if 'attention_mask' in tok and tok['attention_mask'] is not None else None
        ).to(torch.float32).detach()
        
        rh_in_test = arc_embedder(test_in)
        out_sys1 = brain(lh_latents=z_lh_test, rh_inputs=rh_in_test)
        logits_sys1 = arc_head(out_sys1['rh_latents_updated'] + rh_in_test, input_grids=test_in, input_masks=test_m_in)
        pred_sys1 = logits_sys1.argmax(dim=1)[0, :H_out, :W_out].cpu().numpy()
        sys1_acc = (pred_sys1 == gt_grid).mean() * 100.0
    sys1_time = (time.time() - t0) * 1000
    
    conflict = out_sys1['conflict_score'].item()
    urgency = out_sys1['affective_state']['urgency'].item()"""

    new_c2 = """    # -------------------------------------------------------------------
    # Condition 2: Bi-Hemispheric System 1 (Fast Feedforward Reflex)
    # -------------------------------------------------------------------
    brain.load_state_dict(base_state_dict['brain'])
    arc_embedder.load_state_dict(base_state_dict['arc_embedder'])
    arc_head.load_state_dict(base_state_dict['arc_head'])
    
    t0 = time.time()
    tok_d = lh_model.tokenize([task_prompt] * len(task.train_pairs), max_length=128)
    with torch.no_grad():
        tok = lh_model.tokenize([task_prompt], max_length=128)
        z_lh_test = lh_model.extract_residual_stream(
            tok['input_ids'].to(device),
            tok.get('attention_mask', None).to(device) if 'attention_mask' in tok and tok['attention_mask'] is not None else None
        ).to(torch.float32).detach()
        
        rh_in_test = arc_embedder(test_in)
        out_sys1 = brain(lh_latents=z_lh_test, rh_inputs=rh_in_test)
        logits_sys1 = arc_head(out_sys1['rh_latents_updated'] + rh_in_test, input_grids=test_in, input_masks=test_m_in)
        pred_sys1 = logits_sys1.argmax(dim=1)[0, :H_out, :W_out].cpu().numpy()
        sys1_acc = (pred_sys1 == gt_grid).mean() * 100.0
        
        # Measure System 1 Demonstration Fit
        z_lh_demo = lh_model.extract_residual_stream(
            tok_d['input_ids'].to(device),
            tok_d.get('attention_mask', None).to(device) if 'attention_mask' in tok_d and tok_d['attention_mask'] is not None else None
        ).to(torch.float32).detach()
        rh_in_d = arc_embedder(demo_in)
        out_d_s1 = brain(lh_latents=z_lh_demo, rh_inputs=rh_in_d)
        l_d_s1 = arc_head(out_d_s1['rh_latents_updated'] + rh_in_d, input_grids=demo_in, input_masks=demo_m_in)
        preds_d_s1 = l_d_s1.argmax(dim=1)
        s1_demo_fit = float(((preds_d_s1 == demo_out.clamp(0, 9)) * demo_m_out).sum().item()) / float(demo_m_out.sum().item() + 1e-8)
    sys1_time = (time.time() - t0) * 1000
    
    conflict = out_sys1['conflict_score'].item()
    urgency = out_sys1['affective_state']['urgency'].item()"""

    old_c4_and_c5 = """    # -------------------------------------------------------------------
    # Condition 4: Bi-Hemispheric System 2 (Full TTA 30 steps)
    # -------------------------------------------------------------------
    brain.load_state_dict(base_state_dict['brain'])
    arc_embedder.load_state_dict(base_state_dict['arc_embedder'])
    arc_head.load_state_dict(base_state_dict['arc_head'])
    
    t0 = time.time()
    tok_d = lh_model.tokenize([task_prompt] * len(task.train_pairs), max_length=128)
    with torch.no_grad():
        z_lh_demo = lh_model.extract_residual_stream(
            tok_d['input_ids'].to(device),
            tok_d.get('attention_mask', None).to(device) if 'attention_mask' in tok_d and tok_d['attention_mask'] is not None else None
        ).to(torch.float32).detach()
        
    params_sys2 = (
        list(brain.corpus_callosum.parameters()) +
        list(brain.right_hemisphere.parameters()) +
        list(arc_embedder.parameters()) +
        list(arc_head.parameters())
    )
    opt_sys2 = optim.AdamW(params_sys2, lr=3.5e-4, weight_decay=1e-4)
    
    for step in range(1, 31):
        opt_sys2.zero_grad()
        rh_in_d = arc_embedder(demo_in)
        out_d = brain(lh_latents=z_lh_demo, rh_inputs=rh_in_d)
        l_d = arc_head(out_d['rh_latents_updated'] + rh_in_d, input_grids=demo_in, input_masks=demo_m_in)
        loss = (ce_fn(l_d, demo_out.clamp(0, 9)) * demo_m_out).sum() / (demo_m_out.sum() + 1e-8)
        (loss + 0.05 * out_d['callosum_losses']['loss_homeostatic']).backward()
        nn.utils.clip_grad_norm_(params_sys2, 1.0)
        opt_sys2.step()
        
    with torch.no_grad():
        rh_in_t = arc_embedder(test_in)
        out_sys2 = brain(lh_latents=z_lh_test, rh_inputs=rh_in_t)
        logits_sys2 = arc_head(out_sys2['rh_latents_updated'] + rh_in_t, input_grids=test_in, input_masks=test_m_in)
        pred_sys2 = logits_sys2.argmax(dim=1)[0, :H_out, :W_out].cpu().numpy()
        sys2_acc = (pred_sys2 == gt_grid).mean() * 100.0
    sys2_time = (time.time() - t0) * 1000
    
    # -------------------------------------------------------------------
    # Condition 5: Dynamic Amygdalar Router Decision
    # -------------------------------------------------------------------
    is_system_1 = (conflict < 0.25 and urgency < 0.35)
    router_decision = 'System 1 (Reflex)' if is_system_1 else 'System 2 (TTA)'
    router_acc = sys1_acc if is_system_1 else sys2_acc
    router_time = sys1_time if is_system_1 else (sys1_time + sys2_time)"""

    new_c4_and_c5 = """    # -------------------------------------------------------------------
    # Condition 4: Bi-Hemispheric System 2 (Adaptive TTA with Early Stopping)
    # -------------------------------------------------------------------
    brain.load_state_dict(base_state_dict['brain'])
    arc_embedder.load_state_dict(base_state_dict['arc_embedder'])
    arc_head.load_state_dict(base_state_dict['arc_head'])
    
    t0 = time.time()
    params_sys2 = (
        list(brain.corpus_callosum.parameters()) +
        list(brain.right_hemisphere.parameters()) +
        list(arc_embedder.parameters()) +
        list(arc_head.parameters())
    )
    opt_sys2 = optim.AdamW(params_sys2, lr=3.5e-4, weight_decay=1e-4)
    
    best_demo_fit = s1_demo_fit
    best_brain_sd = copy.deepcopy(brain.state_dict())
    best_emb_sd = copy.deepcopy(arc_embedder.state_dict())
    best_head_sd = copy.deepcopy(arc_head.state_dict())
    best_step = 0
    
    for step in range(1, 31):
        opt_sys2.zero_grad()
        rh_in_d = arc_embedder(demo_in)
        out_d = brain(lh_latents=z_lh_demo, rh_inputs=rh_in_d)
        l_d = arc_head(out_d['rh_latents_updated'] + rh_in_d, input_grids=demo_in, input_masks=demo_m_in)
        loss = (ce_fn(l_d, demo_out.clamp(0, 9)) * demo_m_out).sum() / (demo_m_out.sum() + 1e-8)
        (loss + 0.05 * out_d['callosum_losses']['loss_homeostatic']).backward()
        nn.utils.clip_grad_norm_(params_sys2, 1.0)
        opt_sys2.step()
        
        # Track demonstration fit & early stop on convergence
        with torch.no_grad():
            preds_step = l_d.argmax(dim=1)
            cur_fit = float(((preds_step == demo_out.clamp(0, 9)) * demo_m_out).sum().item()) / float(demo_m_out.sum().item() + 1e-8)
            if cur_fit > best_demo_fit:
                best_demo_fit = cur_fit
                best_brain_sd = copy.deepcopy(brain.state_dict())
                best_emb_sd = copy.deepcopy(arc_embedder.state_dict())
                best_head_sd = copy.deepcopy(arc_head.state_dict())
                best_step = step
                if cur_fit >= 0.999:  # Perfect demonstration fit achieved
                    break
                    
    # Restore best parameter snapshot
    brain.load_state_dict(best_brain_sd)
    arc_embedder.load_state_dict(best_emb_sd)
    arc_head.load_state_dict(best_head_sd)
    
    with torch.no_grad():
        rh_in_t = arc_embedder(test_in)
        out_sys2 = brain(lh_latents=z_lh_test, rh_inputs=rh_in_t)
        logits_sys2 = arc_head(out_sys2['rh_latents_updated'] + rh_in_t, input_grids=test_in, input_masks=test_m_in)
        pred_sys2 = logits_sys2.argmax(dim=1)[0, :H_out, :W_out].cpu().numpy()
        sys2_acc = (pred_sys2 == gt_grid).mean() * 100.0
    sys2_time = (time.time() - t0) * 1000
    
    # -------------------------------------------------------------------
    # Condition 5: Reflex-First Cascaded Router with Pareto Safety Fallback
    # -------------------------------------------------------------------
    if s1_demo_fit >= 0.90:
        # Reflex Demonstration-Fit Gating: instant bypass in 65ms
        router_decision = 'System 1 (Reflex Fit >= 90%)'
        router_acc = sys1_acc
        router_time = sys1_time
    elif best_demo_fit < s1_demo_fit:
        # Monotonic Pareto Fallback: TTA degraded demo fit, safely revert to S1!
        router_decision = 'System 1 (Pareto Fallback)'
        router_acc = sys1_acc
        router_time = sys1_time + sys2_time
    else:
        # Synergistic System 2 adaptation took place
        router_decision = f'System 2 (TTA Step {best_step})'
        router_acc = sys2_acc
        router_time = sys1_time + sys2_time"""

    if old_target in src and old_c4_and_c5 in src:
        src = src.replace(old_target, new_c2)
        src = src.replace(old_c4_and_c5, new_c4_and_c5)
        # Convert back to list of lines with \n
        new_lines = [line + '\n' for line in src.split('\n')]
        # Fix trailing newline
        if new_lines and new_lines[-1] == '\n':
            new_lines = new_lines[:-1]
        bench_cell['source'] = new_lines
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Successfully updated benchmark cell in Colab notebook!")
    else:
        print("Could not match exact target strings, check source diff.")

if __name__ == '__main__':
    update_notebook_benchmark()
