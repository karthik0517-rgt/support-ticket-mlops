from transformers import pipeline

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Your support ticket categories
# candidate_labels = [
#     "Billing",
#     "Technical Support",
#     "Account Issue",
#     "General Query",
#     "Feedback"
# ]

candidate_labels = [
    "Billing and payments",
    "Technical support",
    "Account login or access issue",
    "General questions",
    "Feedback and suggestions"
]
# def predict(text: str):
#     result = classifier(text, candidate_labels)
#     return {
#         "label": result["labels"][0],             # Most probable label
#         "score": round(result["scores"][0], 4)    # Confidence
#     }

def predict(text: str):
    result = classifier(text, candidate_labels)
    print(f"\n🧠 Text: {text}")
    print(f"🔍 Prediction: {result['labels'][0]} (Score: {result['scores'][0]})")
    print(f"🧾 Full result: {result}")
    return {
        "label": result["labels"][0],
        "score": round(result["scores"][0], 4)
    }
