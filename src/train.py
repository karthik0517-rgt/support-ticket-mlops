import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from datasets import Dataset
import mlflow
import mlflow.pytorch

# =========================
# Configs
# =========================
MODEL_NAME = "distilbert-base-uncased"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# =========================
# Load Data
# =========================
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Encode string labels to integers
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

# Convert to Hugging Face Dataset
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# =========================
# Tokenization
# =========================
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# =========================
# Load Model
# =========================
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_encoder.classes_))

# =========================
# Metrics Function
# =========================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# =========================
# Training Arguments
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    #evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none"  # disables wandb/huggingface reporting
)

# =========================
# MLflow Setup
# =========================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("support-ticket-classifier")

with mlflow.start_run():
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("epochs", training_args.num_train_epochs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    mlflow.log_metrics(eval_metrics)

    # Save model
    model_path = "models/hf_model"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    # Log model
    mlflow.pytorch.log_model(model, "model")

    print(f"✅ ✅ Training complete. Accuracy: {eval_metrics['eval_accuracy']:.2f}")
