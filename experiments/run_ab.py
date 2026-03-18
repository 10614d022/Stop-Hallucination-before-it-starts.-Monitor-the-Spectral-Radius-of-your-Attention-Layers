import random
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from spectral.probe import compute_token_level_rho, compute_entropy
from spectral.clamp import compute_tau_from_entropy
from spectral.metrics import detect_repetition


def run_experiment(prompt: str, model_name: str = "gpt2", runs: int = 3, max_steps: int = 40,
                   theta: float = 3.5, alpha_map=None):
    if alpha_map is None:
        alpha_map = {"CONTROL": 0.0, "MILD": 0.7, "STRONG": 1.5}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    groups = ["CONTROL", "MILD", "STRONG"]
    results = []

    for seed in range(1, runs + 1):
        torch.manual_seed(seed)
        random.seed(seed)

        for group in groups:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            tokens = input_ids[0].tolist()
            t_rho, t_collapse = -1, -1

            for step in range(max_steps):
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, output_attentions=True)

                final_attn = outputs.attentions[-1][0, 0]  # first head, final layer
                rho = compute_token_level_rho(final_attn)
                ent = compute_entropy(final_attn)

                if ent < theta and t_rho == -1:
                    t_rho = step

                logits = outputs.logits[:, -1, :]
                tau = 1.0 if group == "CONTROL" else compute_tau_from_entropy(ent, theta, alpha_map[group])
                probs = torch.softmax(logits / tau, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                tokens = input_ids[0].tolist()

                if detect_repetition(tokens) and t_collapse == -1:
                    t_collapse = step

            lead_time = (t_collapse - t_rho) if (t_rho >= 0 and t_collapse >= 0) else None
            results.append({
                "group": group,
                "seed": seed,
                "t_rho": t_rho,
                "t_collapse": t_collapse,
                "collapsed": int(t_collapse >= 0),
                "lead_time": lead_time,
                "max_steps": max_steps,
            })

    return pd.DataFrame(results)
