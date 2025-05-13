import modal

app = modal.App("clip-networth-api")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
)

volume = modal.Volume.from_name("selfie-networth-api")
secrets = [modal.Secret.from_name("qdrant-secret")]

@app.function(image=image, secrets=secrets, volumes={"/models": volume})
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse
    from transformers import CLIPModel, CLIPProcessor
    from qdrant_client import QdrantClient
    from PIL import Image, UnidentifiedImageError
    import torch, joblib, numpy as np
    import os, io

    app = FastAPI()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)
    clip_model.eval()

    # ✅ Loads model from persisted Modal volume
    print("Available files in /models:", os.listdir("/models"))
    regressor = joblib.load("/models/models/networth_regressor.pkl")

    # ✅ Initializes Qdrant client with secrets
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"]
    )

    COLLECTION = "celebrity_faces"

    def extract_embedding(image: Image.Image):
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        return emb.squeeze().cpu().numpy()

    @app.post("/predict/")
    async def predict(file: UploadFile = File(...)):
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(400, "Invalid image")
        except Exception:
            raise HTTPException(500, "Image error")

        try:
            embedding = extract_embedding(image)
            net_worth = regressor.predict([embedding])[0]
            net_worth = round(np.expm1(net_worth) / 1e6)

            results = qdrant.search(
                collection_name=COLLECTION,
                query_vector=embedding.tolist(),
                limit=3,
                with_payload=True
            )

            return JSONResponse({
                "estimated_net_worth": f"{net_worth}M (USD)",
                "top_similar_profiles": [
                    {
                        "name": r.payload.get("name", "unknown"),
                        "net_worth": f"{round(r.payload.get('net_worth', 0)/1e6)}M (USD)",
                        "score": round(r.score, 4)
                    } for r in results
                ]
            })
        except Exception:
            raise HTTPException(500, "Prediction failed")

    return app
