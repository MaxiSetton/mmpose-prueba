# api.py

import tempfile
import requests
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

# Import your custom classes
from model import Predictor
from kp_api import KeypointExtractor

# --- Configuration ---
# IMPORTANT: Update these paths to match your file locations
DEVICE = 'cpu'  # or 'cuda' if you have a GPU
PREDICTOR_CFG_PATH = 'phoenix-2014t.yaml'
PREDICTOR_CKPT_PATH = 'best.pth'
KP_EXTRACTOR_CFG = 'rtmw-l_8xb320-270e_cocktail14-384x288.py'  # MMPose config alias or path
KP_EXTRACTOR_CKPT = 'rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth'

# --- Global Model Initialization ---
# This is crucial for performance. Models are loaded once when the API starts.
print("Initializing models...")
app = FastAPI()
kp_extractor = KeypointExtractor(KP_EXTRACTOR_CFG, KP_EXTRACTOR_CKPT, device=DEVICE)
predictor = Predictor(PREDICTOR_CFG_PATH, PREDICTOR_CKPT_PATH, device=DEVICE)
ALPHA = predictor.config['model']['RecognitionNetwork']['SlowFast']['Alpha']
print("Models initialized successfully. API is ready.")


# --- API Endpoint ---
@app.get("/predict_gloss_from_url")
async def predict_gloss_from_url(video_url: str):
    """
    Downloads a video, extracts keypoints, and predicts sign language glosses.
    Example URL: http://127.0.0.1:8000/predict_gloss_from_url?video_url=YOUR_CLOUDINARY_URL
    """
    try:
        # 1. Download video from the URL into a temporary file
        print(f"Downloading video from: {video_url}")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp_file:
            with requests.get(video_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
            tmp_file.seek(0)  # Go back to the beginning of the file

            # 2. Extract keypoints from the downloaded video
            keypoints = kp_extractor.extract_from_video(tmp_file.name)

            if keypoints is None:
                raise HTTPException(status_code=400, detail="Could not detect any person or keypoints in the video.")

            # 3. Pad keypoints to be a multiple of ALPHA
            num_frames = keypoints.shape[0]
            remainder = num_frames % ALPHA
            if remainder != 0:
                padding_needed = ALPHA - remainder
                print(
                    f"Padding keypoints: {num_frames} frames -> {num_frames + padding_needed} frames (multiple of {ALPHA})")
                last_frame_keypoints = keypoints[-1]
                padding = np.repeat(last_frame_keypoints[np.newaxis, :, :], padding_needed, axis=0)
                keypoints = np.vstack([keypoints, padding])

            # 4. Get gloss prediction
            print("Sending keypoints to the predictor...")
            predicted_glosses = predictor(keypoints)

            return {"predicted_glosses": " ".join(predicted_glosses[0])}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video from URL: {e}")
    except Exception as e:
        # Catch-all for other potential errors
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)