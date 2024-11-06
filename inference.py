# -*- coding: utf-8 -*-

# Importing Packages
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import torch


# Function to perform inference using the saved model
def perform_inference(input_texts):
    """Load the trained model and tokenizer, and perform inference on input texts."""
    # Load the trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("trained_model")
    tokenizer = AutoTokenizer.from_pretrained("trained_model")

    # Tokenize the input texts
    inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Perform inference (forward pass through the model)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get predictions (convert logits to class labels)
    predictions = np.argmax(logits.numpy(), axis=-1)

    return predictions


if __name__ == "__main__":
    # Example input texts to classify
    input_texts = ["This is a health-related fact.", "This is a false claim."]

    # Get predictions for the input texts
    predictions = perform_inference(input_texts)

    # Print predictions (numeric labels)
    print("Predictions:", predictions)
