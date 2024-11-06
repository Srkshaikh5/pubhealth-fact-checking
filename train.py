# -*- coding: utf-8 -*-

""" Model Selection
Model Choice: Choose distilbert-base-uncased-finetuned-health_facts as it balances performance and efficiency."""

# Import necessary libraries
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from evaluate import load

# Constants and configurations
MODEL_NAME = "austinmw/distilbert-base-uncased-finetuned-health_facts"
VALID_LABELS = [0, 1, 2, 3]
BATCH_SIZE = 8
EPOCHS = 3
MAX_LENGTH = 512  # adjust based on model input size


# Functions for loading and preprocessing data
def load_and_preprocess_data(train_file, test_file):
    """Load and preprocess the training and testing datasets."""
    # Load datasets
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Filter rows with invalid labels
    train_data = train_data[train_data['label'].isin(VALID_LABELS)]
    test_data = test_data[test_data['label'].isin(VALID_LABELS)]

    # Ensure labels are integers
    train_data['label'] = train_data['label'].astype(int)
    test_data['label'] = test_data['label'].astype(int)

    # Print sample data and columns for debugging
    print("Train Data Columns:", train_data.columns)
    print("Test Data Columns:", test_data.columns)
    print("Sample Training Data:", train_data.head())
    print("Sample Test Data:", test_data.head())

    # Convert DataFrame to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    return train_dataset, test_dataset


# Function for tokenizing the datasets
def tokenize_function(examples, tokenizer):
    """Tokenize the input text using the provided tokenizer."""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)


def tokenize_datasets(train_dataset, test_dataset, tokenizer):
    """Tokenize the training and test datasets."""
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    return train_dataset, test_dataset


# Metric computation function
def compute_metrics(eval_pred):
    """Compute accuracy metric for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Main training script
def train_model(train_file, test_file):
    """Load data, tokenize, and train the model."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load and preprocess datasets
    train_dataset, test_dataset = load_and_preprocess_data(train_file, test_file)

    # Tokenize the datasets
    train_dataset, test_dataset = tokenize_datasets(train_dataset, test_dataset, tokenizer)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

    # Define accuracy metric
    global metric
    metric = load("accuracy")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_dir="logs",
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Save the trained model and tokenizer
    trainer.save_model("trained_model")  # Save model weights and config
    tokenizer.save_pretrained("trained_model")  # Save tokenizer
    print("Model saved to 'trained_model' directory.")


# Runing the training process
if __name__ == "__main__":
    train_file = "data/train_processed.csv"
    test_file = "data/test_processed.csv"

    # Train the model
    train_model(train_file, test_file)
