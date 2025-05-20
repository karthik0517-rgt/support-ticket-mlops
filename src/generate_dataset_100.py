import pandas as pd
import random

categories = ["Billing", "Technical Support", "Account Issue", "General Query", "Feedback"]

examples = {
    "Billing": [
        "I was double charged on my card",
        "Need help with my refund",
        "Why was I charged extra this month?"
    ],
    "Technical Support": [
        "App keeps crashing",
        "I can't upload my documents",
        "App is not responding"
    ],
    "Account Issue": [
        "I can’t access my account",
        "Password reset is not working",
        "Account suspended without reason"
    ],
    "General Query": [
        "How to update my profile?",
        "Do you offer student discounts?",
        "Can I pause my subscription?"
    ],
    "Feedback": [
        "Great customer support!",
        "Loved the new app interface",
        "Service is fast and smooth"
    ]
}

data = []

for category in categories:
    for sentence in examples[category]:
        data.append({"text": sentence, "label": category})

# Add some random duplicates for larger size
for _ in range(100):
    category = random.choice(categories)
    sentence = random.choice(examples[category])
    data.append({"text": sentence, "label": category})

df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

# Split
train = df[:100]
test = df[100:120]

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("✅ Dataset created in 'data/train.csv' and 'data/test.csv'")
