import torch
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import evaluate
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Paths and model configuration
DATA_PATH = 'path_to_data/'
MODEL_NAME = 'FacebookAI/xlm-roberta-large-finetuned-conll03-english' 
CACHE_DIR = 'path_to_cache/'
TRAIN_FILE = f'{DATA_PATH}train.csv'
VAL_FILE = f'{DATA_PATH}val.csv'
TEST_FILE = f'{DATA_PATH}test.csv'

# Define label mapping
LABEL_MAP = {
    'O': 0,
    'derogation': 1,
    'animosity': 2,
    'threatening': 3,
    'hatecrime': 4,
    'comparison': 5,
    'ethnicity': 6,
    'religion': 7,
    'sexual_orientation': 8,
    'gender': 9,
    'disability': 10
}

# Function to load and preprocess data
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df['tokens'] = df['text'].apply(word_tokenize)
    # Assuming 'annotations' column has structured data indicating the span and label for each token
    df['labels'] = df.apply(lambda row: label_tokens(row['tokens'], row['annotations']), axis=1)
    return df[['tokens', 'labels']]

# Token labeling based on spans and labels
def label_tokens(tokens, annotations):
    labels = ['O'] * len(tokens)  # Default label
    for annotation in annotations:
        start, end, label = annotation['start'], annotation['end'], annotation['label']
        for i in range(start, end):
            labels[i] = LABEL_MAP.get(label, 0)  # Apply specific label within the span
    return labels

# Load datasets
train_df = load_and_preprocess(TRAIN_FILE)
val_df = load_and_preprocess(VAL_FILE)
test_df = load_and_preprocess(TEST_FILE)

# Tokenizer and data collation
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Convert dataframes to datasets
def dataframe_to_dataset(df):
    return Dataset.from_pandas(df)

train_dataset = dataframe_to_dataset(train_df)
val_dataset = dataframe_to_dataset(val_df)
test_dataset = dataframe_to_dataset(test_df)

# Model definition
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_MAP),
    id2label={v: k for k, v in LABEL_MAP.items()},
    label2id=LABEL_MAP,
    cache_dir=CACHE_DIR
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=f'{CACHE_DIR}/results/',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {'f1': evaluate.load('f1')(p)}
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(f'{CACHE_DIR}/final_model/')

print("Training completed!")