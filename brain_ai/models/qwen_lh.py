"""
brain_ai.models.qwen_lh: Left Hemisphere Foundation LLM Wrapper for Qwen 2.5.

Grounding & Architecture:
- Phase 2 Scaling: Upgrades Left Hemisphere to Qwen 2.5 (14B / 32B AWQ / 7B / 1.5B).
- Intermediate residual stream hook (default: Layer 24 for 14B/32B or Layer 16).
- Forward injection hook for transcallosal spatial feedback delta_LH.
- Typed Python DSL prompt formatting for inductive ARC-AGI-2 program synthesis.
- Dual-mode: Live Hugging Face model on CUDA or lightweight mock mode for local testing/CI.
"""

from typing import Optional, Tuple, Dict, Any, List
import os
import torch
import torch.nn as nn
import numpy as np


class LeftHemisphereQwen(nn.Module):
    """
    Left Hemisphere Symbolic-Linguistic Engine (Qwen 2.5):
    - Acts as the lateralized linguistic and program synthesis brain.
    - Extracts intermediate residual activations Z_L to send across the Corpus Callosum.
    - Accepts transcallosal spatial feedback delta_LH from the Right Hemisphere.
    - Generates typed ARC DSL program candidates.
    """
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-14B-Instruct",
        hook_layer: int = 24,
        d_model: int = 5120,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mock_mode: bool = False
    ):
        super().__init__()
        self.model_id = model_id
        self.hook_layer = hook_layer
        self.d_model = d_model
        self.target_device = device
        self.torch_dtype = torch_dtype
        self.mock_mode = mock_mode
        
        self.tokenizer = None
        self.model = None
        self._current_injection: Optional[torch.Tensor] = None
        self._injection_consumed: bool = False
        self._hook_handle = None
        
        if not mock_mode:
            self._init_live_model()
        else:
            self._init_mock_model()

    def _init_live_model(self):
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[LH Qwen] Loading tokenizer: {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"[LH Qwen] Loading backbone in {self.torch_dtype} on {self.target_device}...")
            # Detect model dim dynamically if available
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.target_device,
                trust_remote_code=True
            )
            if hasattr(self.model.config, "hidden_size"):
                self.d_model = self.model.config.hidden_size
                
            # Freeze base weights
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
                
            # Adjust hook layer if deeper than available layers
            num_layers = getattr(self.model.config, "num_hidden_layers", 28)
            if self.hook_layer >= num_layers:
                self.hook_layer = num_layers // 2
                
            # Register forward hook on intermediate residual stream
            self._register_residual_hook()
            print(f"[LH Qwen] Successfully initialized and hooked at Layer {self.hook_layer} (d_model={self.d_model})!")
        except Exception as e:
            print(f"[LH Qwen] Warning: Could not load live model '{self.model_id}': {e}")
            if hasattr(self, "model") and self.model is not None:
                del self.model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[LH Qwen] Falling back to Mock Mode for testing/CI.")
            self.mock_mode = True
            self._init_mock_model()

    def _init_mock_model(self):
        """Lightweight simulation for local CI/testing without 14GB checkpoint."""
        self.mock_embed = nn.Embedding(152064, self.d_model)
        self.mock_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 2,
            batch_first=True
        )

    def _register_residual_hook(self):
        if self.model is None or not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            return
            
        target_layer = self.model.model.layers[self.hook_layer]
        
        def hook_fn(module, input_tuple, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            self._last_residual_state = hidden_states
            
            # Inject callosal feedback if queued
            if self._current_injection is not None and not self._injection_consumed:
                inj = self._current_injection
                if inj.shape[-1] != hidden_states.shape[-1]:
                    raise ValueError(f"Injection dim {inj.shape[-1]} != residual dim {hidden_states.shape[-1]}")
                # Broadcast across sequence length if needed
                if inj.shape[1] == 1 and hidden_states.shape[1] > 1:
                    inj = inj.expand(-1, hidden_states.shape[1], -1)
                hidden_states = hidden_states + inj.to(hidden_states.device, dtype=hidden_states.dtype)
                self._injection_consumed = True
                
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                return hidden_states
            return output

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def set_callosal_injection(self, delta_lh: torch.Tensor):
        """Queues transcallosal spatial feedback into the Left Hemisphere residual stream."""
        self._current_injection = delta_lh
        self._injection_consumed = False

    def clear_injection(self):
        self._current_injection = None
        self._injection_consumed = False

    def forward(
        self,
        prompt_text: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Runs a forward pass through the Left Hemisphere.
        Returns:
            'residual_latent': Intermediate representation Z_L (B, S, d_model)
            'logits': Token prediction logits (B, S, V)
        """
        if self.mock_mode:
            if input_ids is None:
                if prompt_text is not None:
                    # Deterministic hash mock tokenization
                    token_vals = [abs(hash(w)) % 10000 + 1 for w in prompt_text.split()][:32]
                    if not token_vals:
                        token_vals = [1, 2, 3]
                    input_ids = torch.tensor([token_vals], dtype=torch.long, device=self.target_device)
                else:
                    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=self.target_device)
                    
            B, S = input_ids.shape
            emb = self.mock_embed(input_ids)
            h = self.mock_layer(emb)
            
            if self._current_injection is not None:
                inj = self._current_injection
                if inj.shape[1] == 1 and S > 1:
                    inj = inj.expand(-1, S, -1)
                h = h + inj[:, :S, :self.d_model]
                self._injection_consumed = True
                
            mock_logits = torch.randn(B, S, 1000, device=input_ids.device)
            return {
                "residual_latent": h,
                "logits": mock_logits,
                "input_ids": input_ids
            }
            
        # Live Model Execution
        if input_ids is None:
            if prompt_text is None:
                prompt_text = "Solve ARC-AGI-2 grid transformation reasoning challenge."
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.target_device)
            attention_mask = encoded["attention_mask"].to(self.target_device)
        else:
            input_ids = input_ids.to(self.target_device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = attention_mask.to(self.target_device)
                
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            
        z_lh = getattr(self, "_last_residual_state", None)
        if z_lh is None:
            raise RuntimeError("Residual hook did not capture intermediate activations.")
            
        return {
            "residual_latent": z_lh,
            "logits": outputs.logits,
            "input_ids": input_ids
        }

    def format_arc_dsl_prompt(
        self,
        demos: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        task_id: str = "unknown"
    ) -> str:
        """
        Formats structured CoT induction prompt requesting typed Python DSL code.
        """
        def grid_to_str(g: np.ndarray) -> str:
            return "\n".join(" ".join(str(int(c)) for c in row) for row in g)

        demo_blocks = []
        for i, (inp, out) in enumerate(demos):
            demo_blocks.append(
                f"### Demonstration {i+1}:\n"
                f"INPUT ({inp.shape[0]}x{inp.shape[1]}):\n{grid_to_str(inp)}\n\n"
                f"OUTPUT ({out.shape[0]}x{out.shape[1]}):\n{grid_to_str(out)}\n"
            )

        test_block = (
            f"### Test Challenge:\n"
            f"INPUT ({test_input.shape[0]}x{test_input.shape[1]}):\n{grid_to_str(test_input)}\n"
        )

        prompt = (
            f"You are the Left Hemisphere of a Bi-Hemispheric Neuromorphic AI solving ARC-AGI-2 task {task_id}.\n"
            f"Synthesize an exact, deterministic Python DSL program that maps each INPUT grid to its OUTPUT grid.\n"
            f"Available DSL primitives: connected_components, bounding_box, crop, paste, rotate90, flip_h, "
            f"flip_v, flood_fill, recolor, gravity.\n\n"
            + "\n".join(demo_blocks)
            + "\n"
            + test_block
            + "\nProvide your reasoning and the Python function `transform(grid: np.ndarray) -> np.ndarray` in a ```python block."
        )
        return prompt
