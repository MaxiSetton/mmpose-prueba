

import torch
import functools

# --- START OF CRITICAL FIX (MONKEY-PATCH) ---
# This block intercepts all calls to `torch.load` and forces `weights_only=False`.
# This is a powerful fix for the stubborn unpickling error.
_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _custom_torch_load
# --- END OF CRITICAL FIX ---


import tempfile
import requests
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

# Import your custom classes
from kp_api import KeypointExtractor  # Make sure this file is named kp_api.py
from model import Predictor

DEVICE = 'cpu'  # or 'cuda' if you have a GPU
PREDICTOR_CFG_PATH = 'phoenix-2014t.yaml'
PREDICTOR_CKPT_PATH = 'best.pth'
KP_EXTRACTOR_CFG = '/home/ubuntu/SignAI-SFS/AI-Setto/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py'  # MMPose config alias or path
KP_EXTRACTOR_CKPT = '/home/ubuntu/SignAI-SFS/AI-Setto/mmpose/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth'
# api.py

# --- Global Model Initialization ---
print("Initializing models...")
app = FastAPI()
# NOTE: The KeypointExtractor now needs to be inside a startup event
# because the monkey-patch must be defined before the library is imported.
# Let's simplify for now and keep it here. The patch should apply.
kp_extractor = KeypointExtractor(KP_EXTRACTOR_CFG, KP_EXTRACTOR_CKPT, device=DEVICE)
predictor = Predictor(PREDICTOR_CFG_PATH, PREDICTOR_CKPT_PATH, device=DEVICE)
ALPHA = predictor.config['model']['RecognitionNetwork']['SlowFast']['Alpha']
print("Models initialized successfully. API is ready.")

@app.get("/")
async def root():
    return {"message": "Sign Language Recognition API is running. Use the /predict endpoint with a video_url parameter."}

@app.get("/predict")
async def predict(video_url: str):
    try:
        print(f"Downloading video from: {video_url}")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp_file:
            with requests.get(video_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
            tmp_file.seek(0)

            keypoints = kp_extractor.extract_from_video(tmp_file.name)

            if keypoints is None:
                raise HTTPException(status_code=400, detail="Could not detect any person or keypoints in the video.")

            num_frames = keypoints.shape[0]
            remainder = num_frames % ALPHA
            if remainder != 0:
                padding_needed = ALPHA - remainder
                print(
                    f"Padding keypoints: {num_frames} frames -> {num_frames + padding_needed} frames (multiple of {ALPHA})")
                last_frame_keypoints = keypoints[-1]
                padding = np.repeat(last_frame_keypoints[np.newaxis, :, :], padding_needed, axis=0)
                keypoints = np.vstack([keypoints, padding])

            print("Sending keypoints to the predictor...")
            predicted_glosses = predictor(keypoints)

            return {"predicted_glosses": " ".join(predicted_glosses[0])}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video from URL: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)