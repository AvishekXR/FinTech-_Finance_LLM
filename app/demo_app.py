"""Three-panel demo: base alone | fine-tuned (precise) | combined (fact + explanation).

Panel 1  base model alone  -> shows it does not know 2025 facts
Panel 2  fine-tuned model  -> the precise, cited figure (closed-book recall)
Panel 3  combined          -> base model explains, grounded in panel 2's fact

One model in memory; the adapter is toggled per answer.

Run on a GPU box (infer venv):
    streamlit run app/demo_app.py --server.port 8501 --server.address 0.0.0.0 \
        --server.fileWatcherType none
"""
import argparse
import datetime
import os
import sys

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "unsloth/Ministral-3-14B-Instruct-2512"

EXPLAIN_PROMPT = """A verified fact from an official source is given below.

Question: {q}
Verified fact: {fact}

Write a clear 3-4 sentence answer for a senior executive. Start from the verified
fact exactly as given (do not change any number), then explain what it means, why
it matters, and any relevant context. Do not invent additional figures."""


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.expanduser("~/jpm_model/adapter"))
    argv = [a for a in sys.argv[1:] if a != "--"]
    args, _ = ap.parse_known_args(argv)
    return args


@st.cache_resource(show_spinner="Loading the 14B model (one time, ~2 min)...")
def load(adapter_path):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE, fix_mistral_regex=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE, device_map={"": 0}, dtype=torch.bfloat16, quantization_config=quant)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            BASE, device_map={"": 0}, dtype=torch.bfloat16, quantization_config=quant)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tok, model


def ask(tok, model, prompt, use_adapter, max_new_tokens=250):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_dict=True,
                                  return_tensors="pt").to(model.device)
    with torch.no_grad():
        if use_adapter:
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            with model.disable_adapter():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


args = get_args()
st.set_page_config(page_title="Private Finance LLM", page_icon="🏦", layout="wide")

st.markdown("""
<style>
  #MainMenu, header, footer {visibility:hidden;}
  .block-container{padding-top:1.4rem;max-width:1280px;}
  html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
  .hero{background:linear-gradient(135deg,#0f2033,#1c3a5e 65%,#1c5fb0 150%);
    border-radius:16px;padding:26px 30px;color:#fff;margin-bottom:20px;
    box-shadow:0 8px 26px rgba(15,32,51,.22);}
  .hero h1{margin:0;font-size:1.8rem;font-weight:800;letter-spacing:-.02em;}
  .hero p{margin:.5rem 0 0;color:#cddcec;font-size:.98rem;max-width:860px;}
  div[data-testid="stForm"]{border:1px solid #e2e8f0;border-radius:14px;
    padding:10px 16px 4px;background:#fff;box-shadow:0 2px 10px rgba(15,32,51,.05);}
  .stButton>button{border-radius:9px;font-weight:600;}
  .stButton>button[kind="primary"]{background:#1c5fb0;border:none;}
  .qline{background:#0f2033;color:#fff;border-radius:10px;padding:11px 18px;
    font-weight:600;margin:22px 0 10px;display:flex;justify-content:space-between;gap:12px;}
  .qline .ts{font-size:.72rem;font-weight:500;color:#8fa8c4;white-space:nowrap;}
  div[data-testid="column"]:nth-of-type(1) div[data-testid="stAlert"]{
    background:#fbeaea;border:1px solid #f3caca;border-radius:12px;}
  div[data-testid="column"]:nth-of-type(2) div[data-testid="stAlert"]{
    background:#e7f5ee;border:1px solid #c9ecd7;border-radius:12px;}
  div[data-testid="column"]:nth-of-type(3) div[data-testid="stAlert"]{
    background:#e8f0fb;border:1px solid #cadcf5;border-radius:12px;}
  div[data-testid="stAlert"] p{color:#1a2532;font-size:.94rem;}
  .cap{font-size:.74rem;font-weight:800;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;}
  .cap.r{color:#a11d24;} .cap.g{color:#0d6b43;} .cap.b{color:#1c5fb0;}
  .sub{font-size:.72rem;color:#6b7a89;font-weight:500;text-transform:none;letter-spacing:0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏦 Private Finance LLM — Precise Recall + Full Explanation</h1>
  <p>One private model, three views. <b>Left:</b> the stock open-source model, which
     does not know 2025 filings. <b>Middle:</b> our fine-tuned model, recalling the exact
     figure from memory with its filing citation. <b>Right:</b> the two working together —
     the verified figure, explained for an executive.</p>
</div>
""", unsafe_allow_html=True)

tok, model = load(args.adapter)

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("ask"):
    q = st.text_input("Ask a financial question:",
                      placeholder="e.g. What was JPMorgan Chase's net income for fiscal 2025?")
    submitted = st.form_submit_button("Ask  →", type="primary")

if submitted and q.strip():
    ts = datetime.datetime.now().strftime("%b %d, %Y · %H:%M:%S")
    with st.spinner("Base model answering..."):
        base_ans = ask(tok, model, q, use_adapter=False)
    with st.spinner("Fine-tuned model recalling the fact..."):
        tuned_ans = ask(tok, model, q, use_adapter=True)
    with st.spinner("Building the explained answer..."):
        combined = ask(tok, model, EXPLAIN_PROMPT.format(q=q, fact=tuned_ans),
                       use_adapter=False, max_new_tokens=320)
    st.session_state.history.insert(0, {"q": q, "ts": ts, "base": base_ans,
                                        "tuned": tuned_ans, "combined": combined})

for turn in st.session_state.history:
    st.markdown(f'<div class="qline"><span>Q&nbsp; {turn["q"]}</span>'
                f'<span class="ts">{turn["ts"]}</span></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="cap r">🔴 Open-source model alone<br>'
                    '<span class="sub">untrained · no 2025 knowledge</span></div>',
                    unsafe_allow_html=True)
        st.info(turn["base"])
    with c2:
        st.markdown('<div class="cap g">🟢 Our fine-tuned model<br>'
                    '<span class="sub">exact figure · cited · from memory</span></div>',
                    unsafe_allow_html=True)
        st.success(turn["tuned"])
    with c3:
        st.markdown('<div class="cap b">🔵 Combined answer<br>'
                    '<span class="sub">verified fact + executive explanation</span></div>',
                    unsafe_allow_html=True)
        st.info(turn["combined"])
