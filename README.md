# 💸 Net Worth Estimator API from Selfie

This project builds an API that estimates a person's net worth from a selfie image. It uses a CLIP image embedding model, a trained regression model, and Qdrant vector search to retrieve the most visually similar celebrity profiles.

---

## 🚀 Features

- Upload a selfie (`multipart/form-data`)
- Predict estimated net worth using a trained regressor on CLIP embeddings
- Retrieve top 3 visually similar celebrity profiles from Qdrant
- Deployed using Docker and ready for local or cloud deployment

---

## 📦 Stack

- [FastAPI](https://fastapi.tiangolo.com/) for serving the API
- [CLIP (ViT-B/32)](https://github.com/openai/CLIP) for image embeddings
- [Qdrant](https://qdrant.tech/) for similarity search
- [Scikit-learn](https://scikit-learn.org/) for regression
- [Docker](https://www.docker.com/) for containerization

---

## 📂 Project Structure

```
.
├── app/
│   ├── main.py                  # FastAPI app
│   └── networth_regressor.pkl   # Trained regression model
├── dataset/
│   ├── celebrity_names_networth.csv  # Celebrity names and net worth
│   └── data_prep.ipynb               # Notebook for data extraction and image scraping
├── prepare_vector_database.py        # Prepare the Qdrant database with embeddings and net worth
├── train_regression_model.ipynb      # Train the regression model using extracted emebddings
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🛠️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/ellie-ei/networth-api.git
cd networth-api
```

### 2. Create conda environtment

```
# Create a new environment
conda create -n networth-env python=3.12

# Activate it
conda activate networth-env

# Install dependencies
pip install -r requirements.txt

# (Optional) Enable Jupyter support
pip install ipykernel
```

### 3. Qdrant Setup

This project expects Qdrant to be running locally via Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Run with Docker

```bash
docker build -t networth-api .
docker run -p 8000:8000 networth-api
```

### 📸 Dataset

- Used 2023 Celebrity Net Worth dataset
- Scraped 1 image per celebrity using DuckDuckGo and the following notebook `data_prep.ipynb`
- Extracted CLIP image embeddings for each image and uploaded all vectors and metadata to Qdrant  using `prepare_vector_database.py`
- Simulated "face-based net worth" by training a regressor on these embeddings using `train_regression_model.ipynb`

### ⚠️ Assumptions

Net worth is not truly predictable from selfies, but this is a fun prototype.
Visual similarity to celebrities is used for similarity search.
The dataset and image scraping are for demonstration only.

### 🧠 Model Details

Embeddings: CLIP (ViT-B/32)

Regression:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Polynomial Regression (degree=2)

Final model was Ridge with log-transformed targets

### 📤 API Endpoint

POST /predict/

Input: Selfie image (multipart/form-data)
Output: JSON with:
estimated_net_worth
top_similar_profiles: 3 most visually similar celebrities

To test the API locally, run

```bash
curl -X POST "http://localhost:8000/predict/" -F "file=@sample_image.jpg"
```
