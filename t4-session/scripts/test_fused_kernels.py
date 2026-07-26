import sys
import torch
from gptqmodel import GPTQModel, BACKEND

DEVICE = "cuda:1"

candidates = [
    ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", BACKEND.EXLLAMA_V2, "GPTQ + EXLLAMA_V2"),
    ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", BACKEND.GPTQ_MARLIN, "GPTQ + GPTQ_MARLIN"),
    ("Qwen/Qwen2.5-0.5B-Instruct-AWQ", BACKEND.AWQ_GEMM, "AWQ + AWQ_GEMM"),
    ("Qwen/Qwen2.5-0.5B-Instruct-AWQ", BACKEND.AWQ_MARLIN, "AWQ + AWQ_MARLIN"),
]

for model_id, backend, label in candidates:
    print(f"\n{'='*70}\nTrying: {label} ({model_id})\n{'='*70}", flush=True)
    try:
        model = GPTQModel.load(model_id, device=DEVICE, backend=backend)
        model = model.model if hasattr(model, "model") else model
        # quick real forward pass to confirm the kernel actually executes, not just loads
        dummy = torch.randint(0, 1000, (1, 16)).to(DEVICE)
        with torch.no_grad():
            out = model(dummy)
        print(f"SUCCESS: {label} loaded AND ran a forward pass. logits shape={out.logits.shape}", flush=True)
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"FAILED: {label} -- {type(e).__name__}: {str(e)[:300]}", flush=True)
        torch.cuda.empty_cache()
