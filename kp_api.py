# keypoint_extractor.py

import cv2
import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model
from tqdm import tqdm

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

class KeypointExtractor:
    def __init__(self, config, checkpoint, device='cpu'):
        """
        Initializes the MMPose model.
        Args:
            config (str): Path to the model config file.
            checkpoint (str): Path to the model checkpoint file.
            device (str): Device to run inference on ('cpu' or 'cuda').
        """
        # This is for a specific numpy version issue with some torch checkpoints
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        self.model = init_model(config, checkpoint, device=device)
        print("Keypoint extractor model loaded successfully.")

    def extract_from_video(self, video_path):
        """
        Extracts whole-body keypoints from a video file.
        Args:
            video_path (str): The local path to the video file.
        Returns:
            np.ndarray: An array of shape (num_frames, 133, 3) with (x, y, confidence)
                        for each keypoint, or None if no poses are detected.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_keypoints = []

        print(f"Extracting keypoints from {total_frames} frames...")
        for _ in tqdm(range(total_frames), desc="Processing Frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # In a real-world scenario, you'd use a person detector here.
            # For this example, we assume one person fills the frame.
            person_bbox = [0, 0, frame_width, frame_height]

            results = inference_topdown(self.model, frame, bboxes=[person_bbox])

            # Ensure a pose was detected
            if not results or not results[0].pred_instances.keypoints.any():
                # If no keypoints, repeat the last frame's keypoints or skip
                if all_keypoints:
                    all_keypoints.append(all_keypoints[-1])
                continue

            # Extract keypoints (x, y) and scores
            keypoints_xy = results[0].pred_instances.keypoints[0]  # Shape: (133, 2)
            scores = results[0].pred_instances.keypoint_scores[0]  # Shape: (133,)

            # Combine into (x, y, score) format
            keypoints_with_scores = np.hstack([keypoints_xy, scores[:, np.newaxis]])
            all_keypoints.append(keypoints_with_scores)

        cap.release()

        if not all_keypoints:
            return None

        return np.array(all_keypoints, dtype=np.float32)