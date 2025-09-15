# in main.py

from model import Predictor
import numpy as np

# Change the number of frames from 10 to 12 (or any multiple of 4)
# Also, keep the dtype from our previous fix!
keypoints = np.zeros(shape=(12, 133, 3), dtype=np.float32)

checkpoint_path = 'best.pth'
device = 'cpu'
cfg_path = 'phoenix-2014t.yaml'

predictor = Predictor(cfg_path, checkpoint_path, device)
glosas = predictor(keypoints)

print(glosas) # Let's print the output to see the result