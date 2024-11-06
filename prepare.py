# -*- coding: utf-8 -*-

# Importing Packages
import pandas as pd


def load_data(file_path):
    """
    Load data from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data


def clean_data(data):
    """
    Clean and preprocess data by handling missing values, duplicates, and encoding labels.
    """
    print("Missing values before dropping:")
    print(data[["claim", "explanation", "label"]].isna().sum())  # Prints count of NaN values per column

    # Drop duplicates based on the 'claim_id' column to avoid repeating claims
    data = data.drop_duplicates(subset="claim_id", keep="first")
    print(f"Shape after dropping duplicates: {data.shape}")

    # Drop rows with missing values in essential columns
    data = data.dropna(subset=["claim", "explanation", "label"])
    print(f"Shape after dropping NaN values: {data.shape}")

    # Retain only the columns needed for training (e.g., claim, explanation, label)
    data = data[["claim", "explanation", "label"]]
    print(f"Shape after selecting relevant columns: {data.shape}")

    # Filter out any rows where label mapping failed
    data = data[data["label"].notna()]
    print(f"Shape after filtering invalid labels: {data.shape}")

    return data


def feature_engineering(data):
    """
    Create a combined text feature by merging 'claim' and 'explanation'.
    """
    # Combine 'claim' and 'explanation' for richer context
    data["text"] = data["claim"] + " " + data["explanation"]
    data = data[["text", "label"]]  # Retain only the features needed for training

    return data


def process_and_save(input_file, output_file):
    """
    Load, clean, process, and save data.
    """
    data = load_data(input_file)
    cleaned_data = clean_data(data)
    processed_data = feature_engineering(cleaned_data)
    processed_data.to_csv(output_file, index=False)

    print("Processed Data")
    print(processed_data.head())
    print(f"Processed data saved to {output_file}.")


if __name__ == "__main__":
    # Process train and test datasets separately
    process_and_save("data/train.csv", "data/train_processed.csv")
    process_and_save("data/test.csv", "data/test_processed.csv")
    print("Data preparation complete for both training and test sets.")
