import os
import torch
import numpy as np
import joblib
import io
import logging
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Globals
clip_model = None
clip_processor = None
regressor = None
qdrant = None
COLLECTION = "celebrity_faces"

@app.on_event("startup")
def load_dependencies():
    global clip_model, clip_processor, regressor, qdrant

    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        clip_model.to(device)
        clip_model.eval()
        app.state.device = device

        regressor = joblib.load("app/networth_regressor.pkl")

        qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
            )

        logger.info("All dependencies loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load dependencies.")
        raise RuntimeError("Startup failure: could not load required models or services.") from e

def extract_embedding(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt").to(app.state.device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb.squeeze().cpu().numpy()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save to in-memory JPEG
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        image = Image.open(buffer)

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unable to process image file.")
    except Exception as e:
        logger.exception("Unexpected error while reading image.")
        raise HTTPException(status_code=500, detail="Internal server error.")

    try:
        embedding = extract_embedding(image)
        net_worth = regressor.predict([embedding])[0]
        net_worth = round(np.expm1(net_worth) / 1000000)  # Inverse of log transformation

        results = qdrant.search(
            collection_name=COLLECTION,
            query_vector=embedding.tolist(),
            limit=3,
            with_payload=True
        )

        top_matches = [
            {
                "name": r.payload.get("name", "unknown"),
                "net_worth": f"{round(r.payload.get("net_worth", 0.0)/1000000)}M (USD)",
                "score": round(r.score, 4)
            } for r in results
        ]

        return JSONResponse({
            "estimated_net_worth": f"{round(net_worth)}M (USD)",
            "top_similar_profiles": top_matches
        })

    except Exception as e:
        logger.exception("Prediction or similarity search failed.")
        raise HTTPException(status_code=500, detail="Failed to process prediction.")