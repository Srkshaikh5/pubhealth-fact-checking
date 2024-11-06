# -*- coding: utf-8 -*-

# Importing Packages
from datasets import load_dataset


def download_data(save_folder="data"):
    # Loading the PUBHEALTH dataset
    dataset = load_dataset("ImperialCollegeLondon/health_fact", trust_remote_code=True)

    # Save the dataset in CSV format
    dataset["train"].to_csv(f"{save_folder}/train.csv", index=False)
    dataset["test"].to_csv(f"{save_folder}/test.csv", index=False)
    print("Dataset downloaded and saved to the data folder.")


if __name__ == "__main__":
    download_data(save_folder="data")
