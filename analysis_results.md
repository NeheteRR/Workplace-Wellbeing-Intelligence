# Workplace Wellbeing Monitoring System - Deep Analysis Report

This document provides a comprehensive, structured overview of the underlying codebase, architecture, AI/ML pipelines, and engineering practices driving the platform.

---

## 1. PROJECT OVERVIEW
- **Problem Solved**: Silent workplace burnout. The system acts as an early warning mechanism for declining employee mental health while preserving individual privacy.
- **Core Objective**: To provide an AI-driven, dual-sided platform that gives employees personalized, conversational mental wellbeing support, while simultaneously feeding anonymized, aggregated risk signals to HR for proactive intervention.
- **Target Users / Use-Case**:
  - **Employees**: Daily journaling/check-ins to track personal emotional wellbeing over time.
  - **HR / Management**: High-level organizational intelligence, burnout trend monitoring, and department-level sentiment distribution tracking.

---

## 2. SYSTEM ARCHITECTURE
- **High-level Architecture**: A decoupled Client-Server architecture. The frontend handles interactive visualizations, while the backend exposes REST APIs wrapped around specialized internal services ([Predictor](file:///d:/Projects_Final/Sentiment_Analysis/backend/services/predictor.py#55-159), [BurnoutEngine](file:///d:/Projects_Final/Sentiment_Analysis/backend/services/wellbeing_trend_engine.py#3-118), [LLMRecommender](file:///d:/Projects_Final/Sentiment_Analysis/backend/services/llm_recommender.py#10-165), and [JsonStore](file:///d:/Projects_Final/Sentiment_Analysis/backend/services/db.py#12-67)).
- **Interaction Flow**:
  1. The **Next.js Frontend** calls the **FastAPI Backend**.
  2. The FastAPI endpoints route business logic to specific isolated classes in `backend/services`.
  3. The `Predictor` loads the heavy TensorFlow `.keras` model, tokenizes the input, and predicts 6 emotional classes.
  4. The `BurnoutEngine` applies heuristic logic over historical logs to determine an overall wellbeing score.
  5. The `LLMRecommender` builds a prompt from these metrics and requests the Google Gemini API for natural-language feedback.
  6. Data is flushed to the `JsonStore`.
- **Deployment Architecture**: Currently configured for local development (`uvicorn` and `next dev`). The codebase lacks Dockerization or production WSGI/ASGI configurations (e.g., gunicorn) in its current state. 

---

## 3. TECH STACK (DETAILED)
- **Programming Languages**: Python 3.9+ (Backend/ML) and TypeScript/Node.js 18+ (Frontend).
- **Frameworks & Libraries**:
  - **Backend**: FastAPI, PyJWT (RBAC Auth), Pydantic (Data validation).
  - **Frontend**: Next.js 15 (App Router), Tailwind CSS v4, Shadcn UI, Recharts, Lucide React.
  - **ML Pipeline**: TensorFlow/Keras, NLTK, Scikit-learn, Pandas.
- **Databases & Storage**: Lightweight `JsonStore` backing data to `data/emotion_logs.json`.
- **AI/ML Models**: 
  - Custom deep learning model (**BiLSTM + Attention Layer**) trained on the *GoEmotions* dataset. Features GloVe (6B.100d) word embeddings.
- **External Services/APIs**: Google Gemini API (`gemini-1.5-flash`) for dynamic conversational suggestions.

---

## 4. END-TO-END FLOW (Employee Check-in)
1. **Input**: Employee submits a daily text log via the Next.js UI (`/analyze-day`).
2. **Preprocessing**: The text is forwarded to the `Predictor`, where it is lowercased, lemmatized via NLTK, scrubbed of stopwords (matching training-time cleaning), and tokenized to a padded sequence.
3. **Inference**: The custom BiLSTM model predicts the probability distribution of 6 emotions (Joy, Sadness, Fear, Anger, Love, Neutral).
4. **Scoring**: `BurnoutEngine` fetches the employee's history. It aggregates "signal days" (where sadness, anger, or fear cross a `0.35` threshold) and applies a persistence penalty to calculate the final `wellbeing_score`.
5. **Generation**: `LLMRecommender` uses the calculated emotions and score to prompt Gemini. It strictly restricts medical advice, returning friendly conversational output and structured suggestions (JSON).
6. **Output**: The combined data is stored centrally and returned as `UserResponse` to the UI to unwrap into charts and an AI chat bubble.

---

## 5. CORE MODULES BREAKDOWN
- `/backend/app.py`: The FastAPI core. Handles JWT authentication, initializes heavy ML singletons, and exposes endpoints for both Employee interfaces (e.g., `/employee/today-summary`) and HR dashboards (e.g., `/org/wellbeing-trend`).
- `/backend/services/predictor.py`: The Deep Learning encapsulation. Reinstantiates the custom Keras `AttentionLayer`. Responsible for text sanitization, tokenization (loaded from `tokenizer.pkl`), and array shape manipulation for tf predictions.
- `/backend/services/wellbeing_trend_engine.py`: The logic brain. Converts raw probability numbers into bounded risk states (`low`, `moderate`, `high`). Implements persistence multipliers for ongoing negative streaks.
- `/backend/services/llm_recommender.py`: The generative facade. Interacts with Gemini API and features a brilliant deterministic fallback system `_fallback_conversational_message` to prevent crashes when the API goes down or hits rate limits.
- `/backend/services/db.py`: A native JSON file parsing abstraction representing the persistence layer.
- `/src/train_model.py`: Training DAG script that loads data, compiles the BiLSTM network with Early Stopping/Checkpoints, pairs Glove definitions, and dumps the final `.keras` and `.pkl` artifacts into `/models`.

---

## 6. AI/ML PIPELINE
- **Data Ingestion**: Processes a balanced CSV of the GoEmotions dataset.
- **Processing**: Complex multi-stage NLP cleaning. Removes URLs/mentions, expands contractions, tags POS for accurate lemmatization, and strips a tailored list of English stopwords while keeping vital negation words (`not`, `never`, `cannot`).
- **Model Architecture**:
  1. Embedding Layer (Initialized with pre-trained GloVe parameters).
  2. Bidirectional LSTM Layer (128 units) capturing contextual dependencies in both directions.
  3. Custom Attention Layer highlighting the most emotionally relevant semantic features in the sequence.
  4. Dense output bounded by Sigmoid activations for multi-class probability.
- **Usage**: Deployed statically as an artifact loaded into RAM on API boot.
- **LLM Reasoning Layer**: Operates entirely downstream of the DL model. It acts purely as a "Translator", converting rigid mathematical outputs (`sadness: 0.65`) into empathetic textual responses.

---

## 7. DESIGN PATTERNS & ENGINEERING PRACTICES
- **Service-Oriented Structure**: The backend code is exceptionally clean. Handlers in `app.py` are thin, routing real complexity into well-named single-purpose classes (`BurnoutEngine`, `Predictor`).
- **Prompt Isolation Rules**: The LLM engineering cleanly prevents hallucinated diagnosis by injecting hard rules (`"Do NOT provide medical advice"`) and dynamic constraints (`allowed_categories`).
- **Scalability Issues**: The system is currently single-node and highly blocking. Deep learning inference runs synchronously on the main thread, and `JsonStore` writes by reading the entire array, appending, and rewriting the entire file synchronously. 

---

## 8. STRENGTHS OF THE SYSTEM
- **Hybrid AI Strategy**: This is an exceptionally smart design. Using a deterministic classification model provides high-speed, controllable, analytical logging while using the LLM strictly as a UI/text-generation tool prevents hallucination-caused data corruption.
- **Fault Tolerance**: The `LLMRecommender` possesses a `_fallback_conversational_message` system. If the Gemini API fails, the backend stays alive and gracefully serves hardcoded empathetic responses.
- **Privacy Focus by Design**: The raw text submissions are passed into the pipeline but are inherently anonymized when calculated into HR dashboard statistics. The `x_source` logic cleanly segregates user viewpoints.

---

## 9. WEAKNESSES / IMPROVEMENTS
- **Failing Database Concurrency (Critical)**: `db.py` -> `_save_all()` overwrites the file simultaneously on any `/analyze-day` call. Under concurrent load, this JSON file *will* get corrupted or overwritten by race conditions. Needs an immediate shift to SQLite or PostgreSQL via SQLAlchemy.
- **Synchronous Bottlenecks**: `predictor.predict()` and Gemini generation block the FastAPI event loop. Heavy models should be executed asynchronously via `async def` wrappers or background workers like Celery/Redis.
- **Hardcoded Secrets**: The JWT key (`dummy_secret_for_jwt`) is publicly visible in `app.py`. Must migrate to `.env`.
- **Model Versioning**: `train_model.py` overwrites the same `best_emotion_model.keras`. A proper ML pipeline (e.g. MLflow) should be added to track model degradation.

---

## 10. RESUME / INTERVIEW SUMMARY
- **Architected a Dual-Dashboard AI Wellbeing Platform** utilizing Next.js, FastAPI, and a custom NLP pipeline to track and predict workplace burnout trends while preserving strictly anonymized organizational intelligence.
- **Designed a Hybrid AI Inference Engine** combining an edge-deployed TensorFlow BiLSTM model (with custom Attention mechanisms) for high-accuracy emotional tracking, backed by an LLM-driven generative layer (Gemini) for dynamic, empathetic user feedback.
- **Integrated Fault-Tolerant Generative Design** by engineering defensive fallbacks for API failure, heavily constraining prompt generation to prevent AI hallucinations or unintended medical diagnostics.
- **Implemented Secure Data Handling and RBAC** exposing decoupled APIs for robust state management, driving complex Recharts data visualizations parsing real-time inference matrices.
