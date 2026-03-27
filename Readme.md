# 🧠 Workplace Wellbeing Intelligence
### Sentiment Analysis & Employee Feedback Monitoring System

## 📌 Project Overview

This system is an AI-driven platform designed to monitor organizational wellbeing while maintaining employee privacy. It leverages a **custom-trained deep learning model** (BiLSTM + Attention) trained on the **GoEmotions dataset** to provide real-world insights into burnout and sentiment trends.

The platform provides a dual-interface experience:
1.  **Employee Dashboard**: Daily emotional check-ins, personal wellbeing tracking, and AI-driven growth suggestions.
2.  **HR Intelligence Dashboard**: Anonymized, aggregated data for organizational health monitoring and early risk signal detection.

---

## 🛠️ Technology Stack

### **Backend (API & AI)**
- **FastAPI**: High-performance Python web framework.
- **TensorFlow/Keras**: Deep learning engine for emotion classification.
- **JWT (PyJWT)**: Secure, stateless authentication and RBAC.
- **JsonStore**: Lightweight persistence layer for sentiment logs.

### **Frontend (Next.js Application)**
- **Next.js 15 (App Router)**: Modern React framework for the dashboard UI.
- **Tailwind CSS & Shadcn/UI**: Premium, responsive UI components.
- **Lucide Icons**: Consistent, high-quality iconography.
- **Recharts**: Interactive data visualizations for wellbeing trends.

---

## 🏗️ Project Architecture

```text
SENTIMENT_ANALYSIS/
├── backend/                # FastAPI Application
│   ├── app.py              # Main API entry point (JWT, Routes)
│   ├── services/           # Business Logic
│   │   ├── predictor.py    # DL Model Inference
│   │   ├── db.py           # JSON Data Management
│   │   ├── wellbeing_trend_engine.py  # Burnout Detection Logic
│   │   └── llm_recommender.py # Conversational AI Responses
│   └── data/               # Persistent Storage (JSON)
├── frontend/               # Next.js Application
│   ├── src/app/            # App Router (Employee & HR routes)
│   ├── src/components/     # Reusable UI Components
│   └── src/lib/            # API Client & Utilities
├── models/                 # ML Artifacts (Keras models, Tokenizers)
├── src/                    # ML Training Pipeline
├── resources/              # Pretrained Embeddings (GloVe)
└── utils/                  # NLP Preprocessing Utilities
```

---

## 🚀 Key Features

### 👤 Employee Experience
- **Daily Check-ins**: Express how you feel through text. The AI analyzes sentiment in real-time.
- **AI Growth Suggestions**: Receive tailored advice based on your current emotional state.
- **Wellbeing History**: Track your progress over time with interactive charts.
- **Privacy First**: Raw text is never shared with management; only aggregated scores are reported.

### 📊 HR & Organizational Intelligence
- **Aggregation Engine**: Automatic calculation of department and organization-level wellbeing scores.
- **Risk Signal Table**: Identify departments or trends that may indicate high burnout risk without compromising individual anonymity.
- **Sentiment Distribution**: Visualize the emotional health of the entire organization (Joy vs Stress vs Neutral).

---

## 🔐 Authentication & Security
The system uses **Role-Based Access Control (RBAC)** powered by JWT.
- **Roles**: `employee` and `hr`.
- **Identity Provider**: Currently uses a mock user database for demonstration (extensible to OAuth or PostgreSQL).

---

## ⚙️ How to Run (Local Development)

### **Prerequisites**
- Python 3.9+
- Node.js 18+
- Active virtual environment (`venv`)

### 1. Start the Backend
```bash
# Set PYTHONPATH to include backend service directory
$env:PYTHONPATH="backend" 
.\venv\Scripts\uvicorn.exe backend.app:app --reload
```

### 2. Start the Frontend
```bash
cd frontend
npm install # Only required once
npm run dev
```

### 3. (Optional) Generate Demo Data
Populate the system with realistic, model-generated historical logs:
```bash
python backend/generate_demo_data_from_model.py
```

---

## ⚖️ Ethical AI & Privacy
- **Anonymization**: Individual feedbacks are processed for trends; raw text inputs are strictly confidential.
- **Bias Mitigation**: Balanced GoEmotions labels (6 core categories) for reliable classification.
- **Supportive Focus**: The tool is designed for resource allocation and support, not for employee surveillance.

---

## 🔮 Roadmap & Future Enhancements
- [ ] **Scalable Data Layer**: Migration from JSON storage to PostgreSQL/MongoDB.
- [ ] **Advanced Forecasting**: Time-series modeling to predict burnout *before* it occurs.
- [ ] **Multilingual Support**: Expanding sentiment analysis to non-English languages.
- [ ] **Containerization**: Full Docker support for seamless cloud deployment.




