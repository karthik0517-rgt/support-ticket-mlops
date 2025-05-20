# 🧠 Support Ticket Classifier using Zero-Shot Learning

This project is an end-to-end **Support Ticket Classifier** API powered by a **Hugging Face zero-shot model** (`facebook/bart-large-mnli`). It classifies incoming support messages into predefined categories like:

- 📄 Billing and Payments  
- 🛠️ Technical Support  
- 🔐 Account Login or Access Issues  
- ❓ General Questions  
- 💬 Feedback and Suggestions

---

## 🚀 Features

- ✅ **Zero-shot classification** — No fine-tuning required
- ⚡ FastAPI backend for real-time prediction
- 🐳 Dockerized for easy deployment
- 📦 Lightweight image (~2.6 GB)
- 🔄 Ready for CI/CD with GitHub Actions
- ☁️ Optionally deployable to AWS, Azure, or GCP

---

## 🧰 Tech Stack

| Layer         | Tech                                |
|---------------|--------------------------------------|
| ML Model      | `facebook/bart-large-mnli` (HF Hub) |
| API           | FastAPI                             |
| Deployment    | Docker                              |
| Versioning    | Git, GitHub                         |
| CI/CD (optional) | GitHub Actions                    |


---

## 📦 Installation & Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/support-ticket-zero-shot.git
cd support-ticket-zero-shot

# 2. Create virtual env
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run locally
uvicorn api.main:app --reload

🐳 Docker Deployment
# Build image
docker build -t support-ticket-zero-shot .

# Run container (with model cache volume)
docker run -p 8000:8000 \
    -v "${PWD}/model_cache:/app/model_cache" \
    -e HF_HOME=/app/model_cache \
    support-ticket-zero-shot

📬 Example Request
http://localhost:8000/docs

POST /predict
{
  "text": "I need a refund for my last payment"
}

Response:
json
{
  "label": "Billing and Payments",
  "score": 0.89
}
