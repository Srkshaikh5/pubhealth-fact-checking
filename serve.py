# -*- coding: utf-8 -*-

# Importing Packages
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

model_name = "austinmw/distilbert-base-uncased-finetuned-health_facts"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class ClaimRequest(BaseModel):
    claim: str


@app.post("/claim/v1/predict")
def predict_veracity(request: ClaimRequest):
    inputs = tokenizer(request.claim, return_tensors="pt")
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    labels = {0: "false", 1: "true", 2: "unproven", 3: "mixture"}
    return {"veracity": labels[prediction]}

# Run: uvicorn main:app --reload
