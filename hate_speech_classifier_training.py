import torch
import pandas as pd
import numpy as np
import argparse
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Tuple

class DatasetLoader:
    def __init__(self, base_path: str = '/path/to/datasets/'):
        self.base_path = base_path
        self.data_directories = {
            'intent': 'intent_spans/',
            'group': 'g_od_spans/'
        }
        self.datasets = {cat: {split: f"{base_path}{self.data_directories[cat]}df_{name}_{cat}_{split}.pkl"
                               for split in ['train', 'val', 'test']
                               for name in ['ihc', 'sbic', 'dh']}
                         for cat in ['intent', 'group']}

    def load_and_process_datasets(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if dataset_name not in ['ihc', 'sbic', 'dh']:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        intent_paths = self.datasets['intent'][dataset_name]
        group_paths = self.datasets['group'][dataset_name]

        def load_dataframe(path: str) -> pd.DataFrame:
            return pd.read_pickle(path).reset_index(drop=True)

        train_df = pd.concat([load_dataframe(intent_paths['train']), load_dataframe(group_paths['train'])])
        val_df = pd.concat([load_dataframe(intent_paths['val']), load_dataframe(group_paths['val'])])
        test_df = pd.concat([load_dataframe(intent_paths['test']), load_dataframe(group_paths['test'])])

        return train_df, val_df, test_df

class HateSpeechClassifier:
    def __init__(self, model_dir: str = '/path/to/models/'):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.special_tokens = [
            'bg_disability', 'eg_disability', 'bg_ethnicity', 'eg_ethnicity',
            'bg_gender', 'eg_gender', 'bg_ideological_group', 'eg_ideological_group',
            'bg_intersection', 'eg_intersection', 'bg_religion', 'eg_religion',
            'bg_sexual_orientation', 'eg_sexual_orientation', 'bg_working_class', 'eg_working_class'
        ]

    def setup_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT",
                                                                        num_labels=2,
                                                                        cache_dir=self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT", cache_dir=self.model_dir)
        self.tokenizer.add_tokens(self.special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def tokenize_data(self, data: pd.DataFrame):
        return self.tokenizer(data['text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_recall_fscore_support(labels, predictions, average='macro')[0],
            'recall': precision_recall_fscore_support(labels, predictions, average='macro')[1],
            'f1': precision_recall_fscore_support(labels, predictions, average='macro')[2]
        }

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate hate speech classifier')
    parser.add_argument('--dataset', type=str, choices=['ihc', 'sbic', 'dh'], required=True, help='Dataset to use')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train', help='Mode of operation')
    parser.add_argument('--model_dir', type=str, default='/path/to/models/', help='Directory for saving models')

    args = parser.parse_args()

    dataset_loader = DatasetLoader()
    classifier = HateSpeechClassifier(model_dir=args.model_dir)

    if args.mode == 'train':
        train_df, val_df, _ = dataset_loader.load_and_process_datasets(args.dataset)
        classifier.setup_model()
        train_encodings = classifier.tokenize_data(train_df)
        val_encodings = classifier.tokenize_data(val_df)
        classifier.train(train_encodings, val_encodings)

    elif args.mode == 'evaluate':
        _, _, test_df = dataset_loader.load_and_process_datasets(args.dataset)
        classifier.evaluate(test_df)

if __name__ == "__main__":
    main()
