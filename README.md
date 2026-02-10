# LLM Inference & Optimization Platform

A production-grade LLM inference platform with:
- Multi-model serving
- Semantic routing
- Caching
- Observability
- Load testing with Locust



A production-grade backend system for Large Language Model inference that demonstrates semantic routing, multi-model orchestration.

This project designed to showcase systems engineering, ML infrastructure design, and performance evaluation similar to internal services used in real-world AI platforms.

---
## Models I have used
I have used Hugging Face MLX models **(specifically optimized to run efficiently on Apple Silicon)**

- **Small model:** [Llama-3.2-3B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit) (1.8GB)
- **Large model:** [Meta-Llama-3.1-8B-Instruct-4bit](https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit) (4.5GB)
---

## Project Highlights

* **Multi-model inference**
* **Semantic routing** using sentence embeddings
* **Clean model abstraction layer** (router and API are model-agnostic)
* **FastAPI backend**
* **Lifecycle-managed model loading (lifespan)**
* **End-to-end latency measurement**
#### Currently working on:
* **Load testing with Locust** using realistic traffic patterns
* **Caching** using redis

---

## 🧠 System Architecture (High Level)

```
Client
  │
  ▼
FastAPI API Layer
  │
  ▼
Semantic Router (Intent + Similarity)
  │
  ├── Small Model (fast, cheap)
  └── Large Model (slow, high-quality)
```

**The API does not know which model is used.** Routing decisions are made dynamically based on semantic similarity.

---

## 📂 Project Structure

```
LLM-Inference-Platform/
│
├── api/
│   ├── main.py          # FastAPI app + lifespan
│   ├── routes.py        # Inference endpoint
│   └── schemas.py       # Request / response / error models
│
├── models/
│   ├── base.py          # Model interface
│   ├── small_model.py   # Lightweight model
│   └── large_model.py   # Large model
│
├── routers/
│   ├── embedder.py      # Sentence embedding model
│   ├── intents.py       # Intent definitions + embeddings
│   └── router.py        # Semantic routing logic
├── tests                # planned
├── cache                # planned
├── observablity         # planned
├── .env.example         # Environment variable template
├── README.md
└── requirements.txt
```

---

## Semantic Routing

Routing is performed using **sentence embeddings**:

1. Prompt → embedding
2. Compare against predefined intent embeddings
3. Compute cosine similarity
4. Route to:

   * **Small model** → simple / casual prompts
   * **Large model** → technical / complex prompts

Routing threshold is configurable via environment variables.

---

## ⚙️ Tech Stack

* **Backend**: FastAPI
* **Models**:
  * Hugging Face Transformers
* **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Routing**: Cosine similarity–based semantic routing
* **Load Testing**: Locust
* **Runtime**: Python 3.11

---

## 🧪 Running the API Locally

### 1️⃣ Create and activate virtual environment

```bash
python -m venv env
source env/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Create `.env` file

```env
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
SMALL_MODEL_NAME=
ROUTING_THRESHOLD=
```

### 4️⃣ Start the API

```bash
uvicorn api.main:app --reload --env-file .env
```

### 5️⃣ Test health endpoint

```bash
curl http://127.0.0.1:8000/health
```

---

## 📡 Inference API

### Endpoint

```
POST /generate
```

### Request

```json
{
  "prompt": "Explain transformers in simple terms",
  "max_tokens": 150
}
```

### Response

```json
{
  "response": "...generated text...",
  "model_used": "large_model",
  "latency_ms": 842.3,
  "intent": "technical",
  "score": 0.71,
  "cached": false
}
```

---

## 🔮 Future Improvements
* Load-Testing with Locust
* Redis-based response caching
* Metrics & observability
* Dockerized deployment

---
## 📜 License

MIT License
