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
- [Modal](https://modal.com) for deployment

---

## 📂 Project Structure

```
.
├── models/
│   └── networth_regressor.pkl         # Trained regression model
├── dataset/
│   ├── celebrity_names_networth.csv  # Celebrity names and net worth
│   └── data_prep.ipynb               # Data extraction and image scraping
├── prepare_vector_database.py        # Populate Qdrant with image embeddings
├── train_regression_model.ipynb      # Train regression model using embeddings
├── upload_model.py                   # Upload model weights to Modal Volume
├── app_modal.py                      # Deploy FastAPI app on Modal
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🛠️ How to Deploy

Modal is used for deployment. Make sure to set up your envrionment with Modal.

### 1. Clone the Repository

```bash
git clone https://github.com/ellie-ei/networth-api.git
cd networth-api
```

### 2. Upload the Model to Modal

This stores the trained regression model in a Modal volume for persistent use.

```bash
modal run upload_model
```

### 3. Deploy the API on Modal

Add your Qdrant API key as secrets in Modal and deploy your app:

```bash
modal secret create qdrant-secret \
  QDRANT_URL=https://your-instance.cloud.qdrant.io \
  QDRANT_API_KEY=your-api-key-here

modal deploy app_modal
```

## Qdrant Setup

This app uses Qdrant as a vector database for similarity search.

### ✅ Steps

1. Create a Qdrant API key and make sure your Qdrant instance is running.

2. Download and extract the dataset:
📁 [Download the images](https://drive.google.com/file/d/1UQVtM-oAUMyOK3z7_hdAWzPp04hjHSKx/view?usp=sharing) and place them in the `dataset/images` folder.

3. Run the following script to extract image embeddings and populate the vector database:

```bash
python prepare_vector_databse.py
```

This script will:

- Load the images
- Generate CLIP embeddings
- Store them in Qdrant with associated metadata (name, net worth)

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
curl -X POST "http://localhost:8000/predict/" -F "file=@Halle_Berry.jpg"
```
