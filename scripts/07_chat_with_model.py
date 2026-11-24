"""Chat with the fine-tuned model (or the plain base model) in the terminal.

Needs: a machine with one NVIDIA GPU (>=16 GB), ~40 GB free disk, and the
inference venv (torch/transformers/peft/bitsandbytes/accelerate). First run
downloads the ~28 GB base model once; later runs load from cache in ~2 min.

Usage:
    python 07_chat_with_model.py --adapter /path/to/adapter   # fine-tuned
    python 07_chat_with_model.py --base-only                  # stock model
Type a question, get an answer (greedy decoding, temperature 0).
Type 'exit' to quit.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Ministral-3-14B-Instruct-2512")
    ap.add_argument("--adapter", default=None, help="path to a trained adapter folder")
    ap.add_argument("--base-only", action="store_true", help="ignore adapter, chat with stock model")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map={"": 0}, dtype=torch.bfloat16,
            quantization_config=quant)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, device_map={"": 0}, dtype=torch.bfloat16,
            quantization_config=quant)

    if args.adapter and not args.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[loaded adapter: {args.adapter}]")
    else:
        print("[stock base model - no adapter]")

    print("Ready. Type a question ('exit' to quit).\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=False)
        print("Model:", tok.decode(out[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip(), "\n")


if __name__ == "__main__":
    main()
