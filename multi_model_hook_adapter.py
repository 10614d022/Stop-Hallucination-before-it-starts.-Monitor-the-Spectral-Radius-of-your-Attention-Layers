import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseAttentionAdapter:
    def __init__(self, model, theta=0.2, alpha=3.0):
        self.model = model
        self.theta = theta
        self.alpha = alpha
        self.current_rho = 0.0
        self.handles = []

    def compute_rho(self, attn_tensor: torch.Tensor) -> float:
        # attn_tensor expected [batch, heads, seq, seq] or [heads, seq, seq]
        if attn_tensor.dim() == 4:
            attn_tensor = attn_tensor[0]
        # head-level proxy on last-token attention
        x = attn_tensor[:, -1, :]
        x = x - x.mean(dim=-1, keepdim=True)
        c = x @ x.transpose(0, 1) / max(x.size(-1), 1)
        v = torch.randn(c.size(0), 1, device=c.device, dtype=c.dtype)
        for _ in range(5):
            v = c @ v
            v = v / (v.norm() + 1e-8)
        return torch.dot(v.squeeze(), (c @ v).squeeze()).item()

    def clamp_output(self, output, rho):
        if rho <= self.theta:
            return output
        tau = 1.0 + self.alpha * (rho - self.theta)
        return output / tau

    def register(self):
        raise NotImplementedError

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class GPT2AttentionAdapter(BaseAttentionAdapter):
    def register(self):
        target = self.model.transformer.h[-1].attn.attn_dropout

        def hook(module, inputs, output):
            rho = self.compute_rho(output)
            self.current_rho = rho
            return self.clamp_output(output, rho)

        self.handles.append(target.register_forward_hook(hook))
        return self


class LlamaAttentionAdapter(BaseAttentionAdapter):
    def register(self):
        target = self.model.model.layers[-1].self_attn

        def hook(module, inputs, output):
            # output may be tuple(hidden_states, attn_weights, present_key_value)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn = output[1]
                rho = self.compute_rho(attn)
                self.current_rho = rho
                return output
            return output

        self.handles.append(target.register_forward_hook(hook))
        return self


class MistralAttentionAdapter(LlamaAttentionAdapter):
    pass


class QwenAttentionAdapter(BaseAttentionAdapter):
    def register(self):
        target = self.model.model.layers[-1].self_attn

        def hook(module, inputs, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn = output[1]
                rho = self.compute_rho(attn)
                self.current_rho = rho
                return output
            return output

        self.handles.append(target.register_forward_hook(hook))
        return self


def build_adapter(model_name: str, model, theta=0.2, alpha=3.0):
    lower = model_name.lower()
    if "gpt2" in lower:
        return GPT2AttentionAdapter(model, theta, alpha).register()
    if "llama" in lower:
        return LlamaAttentionAdapter(model, theta, alpha).register()
    if "mistral" in lower:
        return MistralAttentionAdapter(model, theta, alpha).register()
    if "qwen" in lower:
        return QwenAttentionAdapter(model, theta, alpha).register()
    raise ValueError(f"No adapter defined for model: {model_name}")


if __name__ == "__main__":
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    adapter = build_adapter(model_name, model)
    ids = tok("Paris is the capital of", return_tensors="pt").input_ids
    with torch.no_grad():
        _ = model(ids)
    print({"model": model_name, "rho": adapter.current_rho})
