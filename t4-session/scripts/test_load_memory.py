import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from gptqmodel import GPTQModel, BACKEND

TARGET_ID = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_GPTQ_ID = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"

print("loading target (8-bit bnb) on cuda:0 ...", flush=True)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
target_model = AutoModelForCausalLM.from_pretrained(TARGET_ID, quantization_config=bnb_config, device_map={"": 0})
print(f"target loaded OK. mem allocated cuda:0 = {torch.cuda.memory_allocated(0)/1e9:.2f}GB, reserved = {torch.cuda.memory_reserved(0)/1e9:.2f}GB", flush=True)

print("loading GPTQ draft model on cuda:1 (backend=TORCH, skip Marlin kernel) ...", flush=True)
draft_model = GPTQModel.load(DRAFT_GPTQ_ID, device="cuda:1", backend=BACKEND.TORCH)
draft_model = draft_model.model if hasattr(draft_model, "model") else draft_model
print(f"draft loaded OK. mem allocated cuda:1 = {torch.cuda.memory_allocated(1)/1e9:.2f}GB, reserved = {torch.cuda.memory_reserved(1)/1e9:.2f}GB", flush=True)

target_tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_GPTQ_ID)

print("running a quick generate() with assistant_model to confirm assisted generation works cross-process-free...", flush=True)
inputs = target_tokenizer("Hello, how are you today?", return_tensors="pt").to(target_model.device)
with torch.no_grad():
    out = target_model.generate(**inputs, max_new_tokens=20, do_sample=False,
                                 assistant_model=draft_model, tokenizer=target_tokenizer,
                                 assistant_tokenizer=draft_tokenizer)
print("SUCCESS:", target_tokenizer.decode(out[0], skip_special_tokens=True), flush=True)
