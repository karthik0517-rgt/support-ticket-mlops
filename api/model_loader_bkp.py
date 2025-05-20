from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline
import torch

model_path = "models/hf_model"

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Define the label map (ensure this order matches the one from training)
label_map = {
    "LABEL_0": "Billing",
    "LABEL_1": "Technical Support",
    "LABEL_2": "Account Issue",
    "LABEL_3": "General Query",
    "LABEL_4": "Feedback"
}

# Build pipeline
text_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def predict(text: str):
    result = text_classifier(text)[0]
    label = result["label"]
    readable_label = label_map.get(label, label)
    return {
        "label": readable_label,
        "score": round(result["score"], 4)
    }
