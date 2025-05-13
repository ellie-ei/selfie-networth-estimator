import os
import modal
from pathlib import Path


MODEL_DIR = Path("/models")

app = modal.App("cache-model-weights")
volume = modal.Volume.from_name("selfie-networth-api", create_if_missing=True)

@app.local_entrypoint()
def main():
    model_file = "models/networth_regressor.pkl"
    modal_volume_path = "/models"

    # Run the modal function
    with volume.batch_upload() as batch:
        batch.put_file(model_file, f"{modal_volume_path}/{os.path.basename(model_file)}")