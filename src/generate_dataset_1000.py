import pandas as pd
import random

# Define categories
categories = {
    "Billing": [
        "I was charged twice",
        "Please refund the amount",
        "Why was I billed extra?",
        "I want my money back",
        "There is an error in my invoice",
        "My payment didn't go through properly",
        "Double charge on credit card",
        "Need help with billing mistake",
        "My bill seems incorrect",
        "I got a late fee I shouldn't have"
    ],
    "Technical Support": [
        "The app keeps crashing",
        "I can't upload my ID",
        "Login doesn't work",
        "I’m unable to reset my password",
        "The page is not loading",
        "App shows a blank screen",
        "Mic is not detected during call",
        "Getting an error while submitting form",
        "App is very slow",
        "Camera access is denied"
    ],
    "Account Issue": [
        "My account was suspended",
        "I can't access my profile",
        "Please reactivate my account",
        "I forgot my login email",
        "Account locked after multiple attempts",
        "I didn’t get my OTP",
        "Can't update my phone number",
        "Profile info isn’t saving",
        "I want to delete my account",
        "Change email not working"
    ],
    "General Query": [
        "How can I change my subscription?",
        "Do you offer a free trial?",
        "Can I upgrade anytime?",
        "Tell me about your services",
        "Do you have a mobile app?",
        "What payment methods do you support?",
        "Is there a cancellation fee?",
        "What is your refund policy?",
        "Can I get an invoice copy?",
        "Do you have student discounts?"
    ],
    "Feedback": [
        "Great service!",
        "I love the new update",
        "Awesome support team",
        "The app UI is clean and smooth",
        "Really happy with your features",
        "Thanks for fixing the bugs",
        "Good experience overall",
        "Nice customer interaction",
        "Quick resolution!",
        "App performance has improved a lot"
    ]
}

# Generate 1000 samples (200 per class)
data = []
for label, examples in categories.items():
    for _ in range(200):
        sentence = random.choice(examples)
        # Add random slight variation
        if random.random() > 0.5:
            sentence += random.choice(["", ".", "!", " please", " asap", " now", " urgently"])
        data.append({"text": sentence, "label": label})


# Shuffle and save
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)

# # Print label index mapping
# label2id = {label: idx for idx, label in enumerate(sorted(df["label"].unique()))}
# print("✅ Label to ID mapping used in training:")
# print(label2id)

df.to_csv("data/train.csv", index=False)

print("✅ Dataset created with 1000 rows in 'data/train.csv'")
