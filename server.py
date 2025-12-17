"""
Simple FastAPI translation server hosting mBART (fb/mbart-large-50).

Usage (on Windows machine with model & GPU recommended):
1. Create venv and install deps:
   python -m pip install fastapi uvicorn transformers torch sentencepiece

2. Run server:
   uvicorn server:app --host 0.0.0.0 --port 8000

3. Call from client:
   POST http://<WIN_IP>:8000/translate  {"text":"...","src_lang":"en_XX","tgt_lang":"ja_XX"}

This file is intended to be placed on the Windows machine you control (or a VM). Adjust
`MODEL_ID` to another model if you prefer (e.g. a distilled or smaller model for CPU).
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="mBART Translation Server")

MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model {MODEL_ID} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)


class TranslateReq(BaseModel):
    text: str
    src_lang: Optional[str] = "en_XX"
    tgt_lang: Optional[str] = "ja_XX"
    max_length: Optional[int] = 256


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model": MODEL_ID}


@app.post("/translate")
def translate(req: TranslateReq):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    # Prepare tokenizer/model inputs
    try:
        inputs = tokenizer(
            req.text, return_tensors="pt", truncation=True, max_length=1024
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tokenization failed: {e}")

    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    if input_ids is None:
        raise HTTPException(status_code=500, detail="tokenizer returned no input_ids")

    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # If tokenizer supports language codes, set forced BOS token
    forced_kwargs = {}
    if hasattr(tokenizer, "lang_code_to_id") and req.tgt_lang in getattr(tokenizer, "lang_code_to_id", {}):
        forced_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id[req.tgt_lang]

    try:
        outs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=req.max_length or 256,
            num_beams=4,
            early_stopping=True,
            **forced_kwargs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")

    text = tokenizer.decode(outs[0], skip_special_tokens=True)
    return {"text": text}
