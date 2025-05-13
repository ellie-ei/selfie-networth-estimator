import logging
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "celebrity_faces"
IMAGE_DIR = Path("dataset/images")
DATA_CSV_PATH = Path("dataset/celebrity_names_networth.csv")

def load_clip_model():
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        model.eval()
        logger.info("CLIP model and processor loaded successfully.")
        return model, processor
    except Exception as e:
        logger.exception("Failed to load CLIP model or processor.")
        raise RuntimeError("CLIP model initialization failed.") from e

def get_embedding(image_path: Path, model, processor) -> np.ndarray:
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        return features.squeeze().numpy()
    except UnidentifiedImageError:
        raise ValueError(f"Image at {image_path} is not a valid image.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract embedding from {image_path}: {e}")

def main():
    logger.info("Starting embedding upload process.")

    # Load model and dataset
    clip_model, clip_processor = load_clip_model()

    if not DATA_CSV_PATH.exists():
        logger.error(f"CSV file not found: {DATA_CSV_PATH}")
        return

    df = pd.read_csv(DATA_CSV_PATH)
    logger.info(f"Loaded dataset with {len(df)} rows.")

    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created.")
    except Exception as e:
        logger.warning(f"Collection creation skipped or failed: {e}")

    # Prepare and upload embeddings
    points = []
    for idx, row in df.iterrows():
        name = row.get("Celebrity")
        net_worth = row.get("Estimated Net Worth")

        if pd.isna(name) or pd.isna(net_worth):
            logger.warning(f"Skipping row {idx}: missing data.")
            continue

        img_path = IMAGE_DIR / f"{name.replace(' ', '_')}.jpg"
        if not img_path.exists():
            logger.warning(f"Image not found for {name}: {img_path}")
            continue

        try:
            embedding = get_embedding(img_path, clip_model, clip_processor)
            payload = {"name": name, "net_worth": float(net_worth)}
            point = PointStruct(id=idx, vector=embedding.tolist(), payload=payload)
            points.append(point)
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"✅ Inserted {len(points)} celebrity embeddings into Qdrant.")
    else:
        logger.warning("No embeddings to insert.")

if __name__ == "__main__":
    main()
