# Sentiment Analysis & Wellbeing Monitoring System

## 📌 Project Overview

This project was built with the primary objective of **designing and training a deep learning–based emotion recognition model from scratch** using the **GoEmotions dataset (Hugging Face)** and then **applying the trained model to a real-world wellbeing and burnout monitoring use case**.

The work is divided into two clear phases:

1. **Emotion Model Development**
   - Dataset preprocessing and balancing
   - Custom deep learning architecture design
   - Model training using pretrained word embeddings
2. **Applied Use Case**
   - Daily employee wellbeing check-ins
   - Burnout trend detection
   - Organization-level wellbeing aggregation

This project emphasizes **practical deep learning concepts**, **clean preprocessing pipelines**, and **ethical application of sentiment analysis**.


## 🎯 Objectives

- Build an **emotion classification model from scratch**
- Use **GoEmotions dataset** and reduce it to meaningful emotion groups
- Apply **advanced NLP techniques**:
  - Tokenization
  - Padding
  - GloVe embeddings
  - BiLSTM + Attention
- Design an **end-to-end system**:
  - Model → API → User UI → Organization Dashboard



## 🧠 Emotion Categories Used

From the original GoEmotions labels, emotions were merged into **6 final categories**:

- Sadness  
- Love  
- Joy  
- Anger  
- Fear  
- Neutral  

This reduction was done to improve interpretability and practical applicability.


## 🗂️ Project Structure
```text
SENTIMENT_ANALYSIS/
│
├── backend/
│ ├── app.py # FastAPI backend
│ ├── seed_dummy_data.py
│ ├── streamlit_app.py
│ ├── streamlit_employee.py
│ ├── streamlit_org.py
│ │
│ ├── services/
│ │ ├── db.py # JSON-based storage
│ │ ├── predictor.py # Model inference logic
│ │ ├── llm_recommender.py # Response generation
│ │ ├── wellbeing_trend_engine.py
│ │ ├── schemas.py
│ │ └── init.py
│ │
│ └── data/
│ └── emotion_logs.json
│
├── data/
│ ├── emotion_logs.json
│ ├── goemotions_bal_removing_stopwords.csv
│ └── raw/
│ ├── goemotions_1.csv
│ ├── goemotions_2.csv
│ └── goemotions_3.csv
│
├── models/
│ ├── best_emotion_model.keras
│ ├── best_emotion_model.h5
│ ├── tokenizer.pkl
│ └── config.json
│
├── resources/
│ └── glove/
│ └── glove.6B.100d.txt
│
├── src/
│ ├── train_model.py
│ ├── text_preprocessing.py
│ └── data_preprocessing.py
│
├── utils/
│ ├── attention.py
│ └── preprocess_pipeline.py
│
├── .env
├── venv/
└── README.md
```


## 🔄 Data Preprocessing Pipeline

### Steps Performed

1. **Merge GoEmotions files**
   - `goemotions_1.csv`
   - `goemotions_2.csv`
   - `goemotions_3.csv`

2. **Emotion Grouping**
   - Multiple fine-grained emotions merged into 6 core emotions

3. **Text Cleaning**
   - Lowercasing
   - Contraction expansion
   - Regex cleaning
   - Stopword handling

4. **Class Balancing**
   - Neutral emotion downsampled
   - Over-represented joy samples reduced

5. **Final Output**
   - Balanced dataset saved as:
     ```
     data/goemotions_bal_removing_stopwords.csv
     ```


## 🧠 Model Architecture

**Deep Learning Model Used:**

- Embedding Layer (GloVe – 100d)
- Bidirectional LSTM (128 units)
- Custom Attention Layer
- Fully Connected Dense Layers
- Sigmoid activation (multi-label classification)

### Why This Architecture?

- BiLSTM captures context from both directions
- Attention helps focus on emotionally relevant words
- Sigmoid allows multi-emotion probability output


## 🏋️ Model Training

### Key Details

- Optimizer: `Adam (1e-4)`
- Loss: `Binary Crossentropy`
- Epochs: `15`
- Batch Size: `64`
- Early Stopping & Model Checkpoint used

### Saved Artifacts

- `best_emotion_model.keras`
- `best_emotion_model.h5`
- `tokenizer.pkl`
- `config.json`


## 🚀 Applied Use Case: Wellbeing Monitoring

### Employee Side
- Daily emotional check-in
- Emotion probabilities returned
- Supportive conversational feedback
- Optional wellbeing suggestions

### Organization Side
- Average wellbeing score
- Overall organizational status
- Number of employees tracked
- Privacy-preserving aggregation (no raw text shown)


## 🖥️ Backend & UI

- **Backend**: FastAPI
- **Storage**: JSON-based persistence
- **Employee UI**: Streamlit chat interface
- **Organization UI**: Streamlit dashboard


## ⚖️ Ethical Considerations

- No raw employee text exposed to organization
- Only aggregated wellbeing metrics shown
- Designed for **support**, not surveillance
- Model outputs treated as **signals**, not diagnoses


## 🛠️ How to Run (Local)

### Backend
```bash
uvicorn backend.app:app --reload
```

Employee App
```bash
streamlit run backend/streamlit_employee.py
```

Organization Dashboard
```bash
streamlit run backend/streamlit_org.py
```

📌 Key Takeaways
- Built a deep learning NLP model from scratch
- Applied real ML concepts, not just APIs
- Designed a full ML → Product pipeline
- Focused on interpretability, ethics, and usability

## 🚀 Future Enhancements

- Integrate a real database (PostgreSQL / MongoDB) instead of JSON storage for scalability

- Add secure authentication & role-based access (Employee vs Organization)

- Build a production-grade frontend using React / Next.js for better UX

- Introduce time-series trend modeling for long-term wellbeing prediction

- Enhance the LLM layer with fine-tuned prompts per industry/domain

- Add explainability dashboards for organizations (emotion trends, signals)

- Implement privacy-preserving analytics (aggregation, anonymization)

- Support multilingual emotion analysis

- Enable real-time notifications for critical wellbeing trends

- Deploy using Docker + CI/CD pipeline on cloud (AWS/GCP/Azure)




