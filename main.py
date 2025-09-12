from model import Predictor


#Conseguir los keypoints a partir de un video con shape (frames, 133,3)

#De ejemplo
import numpy as np
keypoints= np.zeros(shape=(10,133,3))
checkpoint_path='best.pth'
device='cpu'
cfg_path='phoenix-2014t.yaml'
predictor= Predictor(cfg_path,checkpoint_path,device)
glosas=predictor(keypoints)