import torchvision
from torchvision.datasets import Kinetics
dataset = Kinetics(root="data", frames_per_clip=1, num_classes='600',split="train", download=True)