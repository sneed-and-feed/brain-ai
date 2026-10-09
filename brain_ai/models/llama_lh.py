"""
brain_ai.models.llama_lh: Left Hemisphere Foundation LLM Wrapper (Llama 3.1 8B).

Provides:
1. Hookable residual stream extraction at intermediate layers (default: Layer 16).
2. Forward injection mechanism: adds transcallosal modulation delta_LH directly into the residual stream.
3. Dual-mode support: Live Hugging Face model on GPU (A100) or lightweight mock mode for local testing/CI.
"""

from typing import Optional, Tuple, Dict, Any, List
import os
import torch
import torch.nn as nn


class LeftHemisphereLlama(nn.Module):
    """
    Left Hemisphere Symbolic-Linguistic Engine:
    Wraps Llama 3.1 8B (or compatible causal LLM) to act as the lateralized linguistic brain.
    Extracts intermediate residual activations Z_L to send across the Corpus Callosum,
    and accepts transcallosal spatial feedback delta_LH from the Right Hemisphere.
    """
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        hook_layer: int = 16,
        d_model: int = 4096,
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
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[LH Llama] Loading tokenizer: {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"[LH Llama] Loading backbone in {self.torch_dtype} on {self.target_device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.target_device,
                trust_remote_code=True
            )
            # Freeze base weights
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
                
            # Register forward hook on intermediate residual stream
            self._register_residual_hook()
            print(f"[LH Llama] Successfully initialized and hooked at Layer {self.hook_layer}!")
        except Exception as e:
            print(f"[LH Llama] Warning: Could not load live model '{self.model_id}': {e}")
            print("[LH Llama] Falling back to Mock Mode for testing.")
            self.mock_mode = True
            self._init_mock_model()

    def _init_mock_model(self):
        """Lightweight simulation for local CI/testing without 16GB checkpoint."""
        self.mock_embed = nn.Embedding(32000, self.d_model)
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
            # Modern transformers return hidden_states Tensor directly; older return tuple
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output
            
            if self._current_injection is not None and not self._injection_consumed and isinstance(hidden_states, torch.Tensor):
                delta = self._current_injection.to(hidden_states.device, dtype=hidden_states.dtype)
                if delta.dim() == 2:
                    delta = delta.unsqueeze(0)
                    
                if hidden_states.dim() == 3:
                    B, S, D = hidden_states.shape
                    # Gentle neuromodulatory gain: clamp callosal perturbation to <= 15% of layer activation norm
                    h_norm = hidden_states.norm(dim=-1, keepdim=True).mean().clamp(min=1.0)
                    d_norm = delta.norm(dim=-1, keepdim=True) + 1e-6
                    scaled_delta = (delta / d_norm) * torch.clamp(d_norm, max=h_norm * 0.15)
                    
                    if scaled_delta.shape[1] == S:
                        hidden_states = hidden_states + scaled_delta
                    elif scaled_delta.shape[1] < S:
                        hidden_states[:, -scaled_delta.shape[1]:, :] = hidden_states[:, -scaled_delta.shape[1]:, :] + scaled_delta
                    else:
                        hidden_states = hidden_states + scaled_delta[:, :S, :]
                    self._injection_consumed = True
                    
                if is_tuple:
                    return (hidden_states,) + output[1:]
                return hidden_states
                
            return output

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def tokenize(self, texts: List[str], max_length: int = 256) -> Dict[str, torch.Tensor]:
        if self.mock_mode or self.tokenizer is None:
            # Deterministic mock tokenization
            B = len(texts)
            input_ids = torch.randint(10, 500, (B, min(32, max_length)), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids.to(self.target_device), "attention_mask": attention_mask.to(self.target_device)}
            
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return {k: v.to(self.target_device) for k, v in enc.items()}

    def extract_residual_stream(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Runs the forward pass up to the intermediate hook layer.
        Returns: Z_L [B, S, d_model]
        """
        if self.mock_mode or self.model is None:
            emb = self.mock_embed(input_ids.to(self.target_device))
            h = self.mock_layer(emb)
            return h

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            # hidden_states is tuple of (embedding_out, layer_1, ..., layer_N)
            # Index hook_layer + 1 corresponds to output of target layer
            z_lh = outputs.hidden_states[self.hook_layer + 1]
            return z_lh

    def generate_with_callosal_feedback(
        self,
        prompt: str,
        delta_lh: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7
    ) -> str:
        """
        Generates text tokens while injecting transcallosal feedback delta_LH
        from the Right Hemisphere into the intermediate residual stream.
        """
        if self.mock_mode or self.model is None or self.tokenizer is None:
            return f"[Simulated LH Llama response given RH feedback (norm: {delta_lh.norm().item() if delta_lh is not None else 0.0:.2f})]"

        self._current_injection = delta_lh
        self._injection_consumed = False
        try:
            inputs = self.tokenize([prompt])
            input_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.05),
                    pad_token_id=self.tokenizer.pad_token_id
                )
            generated_text = self.tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)
            return generated_text.strip()
        finally:
            self._current_injection = None
            self._injection_consumed = False
