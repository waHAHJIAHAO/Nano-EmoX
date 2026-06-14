import os
import sys
from typing import List, Dict, Any

# Ensure the parent directory is in sys.path so 'nano_emox' package can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nano_emox.evaluation.ew_metric import (
    extract_openset_batchcalling,
    func_read_batch_calling_model,
)


class ExtractRequest(BaseModel):
    texts: List[str]


class ExtractResponse(BaseModel):
    emotions: List[str]
    raw_outputs: List[str]


app = FastAPI(title="Nano-EmoX Emotion Extractor", version="1.0.0")

MODEL_NAME = os.environ.get("EMOTION_EXTRACT_MODEL", "Qwen25_7B")

_llm = None
_tokenizer = None
_sampling_params = None


def _to_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        result = []
        for i in x:
            s = str(i).strip().lower()
            s = s.strip("[]").strip("()").strip("{}")
            if s:
                result.append(s)
        return result
    s = str(x).strip()
    if not s:
        return []
    s = s.replace(";", ",")
    s = s.strip("[]").strip("()").strip("{}")
    items = [t.strip().lower() for t in s.split(",") if t.strip()]
    return items


@app.on_event("startup")
def startup_event():
    global _llm, _tokenizer, _sampling_params
    _llm, _tokenizer, _sampling_params = func_read_batch_calling_model(modelname=MODEL_NAME)


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    if _llm is None or _tokenizer is None or _sampling_params is None:
        raise HTTPException(status_code=500, detail="Extractor model is not initialized.")

    if len(req.texts) == 0:
        return ExtractResponse(emotions=[], raw_outputs=[])

    try:
        # print(f"\n[SERVICE /extract] Received {len(req.texts)} texts for extraction")
        for i, text in enumerate(req.texts):
            print(f"[SERVICE /extract] text[{i}] (len={len(text)}): {text[:300]}...")

        name2reason = {f"tmp_{i}": (t if isinstance(t, str) and t.strip() else "neutral") for i, t in enumerate(req.texts)}
        names, outputs = extract_openset_batchcalling(
            name2reason=name2reason,
            llm=_llm,
            tokenizer=_tokenizer,
            sampling_params=_sampling_params,
        )

        idx2out = {n: o for n, o in zip(names, outputs)}
        ordered_raw = [idx2out.get(f"tmp_{i}", "") for i in range(len(req.texts))]
        emotions = ordered_raw

        return ExtractResponse(emotions=emotions, raw_outputs=ordered_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("EMOTION_EXTRACT_HOST", "0.0.0.0")
    port = int(os.environ.get("EMOTION_EXTRACT_PORT", "18081"))
    uvicorn.run(app, host=host, port=port)

