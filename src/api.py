import os
import json
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast

from src.model import load_finetuned_model


class PredictRequest(BaseModel):
    texts: List[str]
    threshold: Optional[float] = 0.5  # threshold for multi-label selection


class PredictResponse(BaseModel):
    predictions: List[Dict[str, float]]


app = FastAPI(title="DistilBERT Multi-label Classifier API")

MODEL_PATH = os.getenv("MODEL_PATH", default=os.path.join(os.path.dirname(__file__), "../output/best_model"))


@app.on_event("startup")
def startup_event():
    # load id2label
    global id2label
    global model
    global tokenizer

    # load model
    try:
        model = load_finetuned_model(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")
    
    # Setting id2label after loading the model
    id2label = model.config.id2label

    # load tokenizer
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    except Exception:
        # try loading by model name/path
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize
    try:
        enc = tokenizer(request.texts, padding=True, truncation=True, return_tensors="pt")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization error: {e}")

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits)

    probs = probs.cpu()

    results = []
    num_labels = probs.size(1)

    for i in range(probs.size(0)):
        row = probs[i]
        class_probs = {}
        predicted_classes = []
        for j in range(num_labels):
            label_name = id2label.get(str(j), id2label.get(j, str(j)))
            prob = float(row[j].item())
            class_probs[label_name] = prob
            if prob >= (request.threshold if request.threshold is not None else 0.5):
                predicted_classes.append(label_name)
        results.append({
            "class_probabilities": class_probs,
            "predicted_classes": predicted_classes
        })

    return {"results": results}


@app.get("/info")
def info() -> Dict[str, Any]:
    # basic model info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "model_path": MODEL_PATH,
        "id2label": id2label,
        "device": device,
        "num_labels": len(id2label) if "id2label" in globals() else None,
    }
