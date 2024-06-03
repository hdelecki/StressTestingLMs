import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset, load_metric
from nltk.tokenize import word_tokenize

# Function to evaluate self-bleu score for a list of generations
def selfbleu(text):
    scores = []
    metric = load_metric("bleu")
    for i in tqdm(range(len(text))):
        references = text[:i] + text[i+1:]
        tokenized_references = [word_tokenize(ref) for ref in references]
        tokenized_prediction = word_tokenize(text[i])
        score = metric.compute(predictions=[tokenized_prediction], references=[tokenized_references])
        scores.append(score["bleu"])
    return np.mean(scores)


# Test script
if __name__ == "__main__":
    # Load the dataset
    model_name = "gpt2"
    data_path = f"./datasets/{model_name}_toxicity_answers"
    dataset = load_dataset(data_path)
    data_list = list(dataset['train']['question'][:1000])
    print(selfbleu(data_list))
