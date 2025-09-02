import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class_names = ["truck", "convertible", "coupe", "hatchback", "SUV", "sedan", "van", "wagon"]

class ResNet18_64(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        
        # Modify first conv for 64x64 input
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        
        # Replace classifier head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

cnn_model = ResNet18_64(num_classes=8)
cnn_model.load_state_dict(torch.load("ResNet18_19e_73a_state", map_location="cpu"))
print("model loaded")
cnn_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = cnn_model.to(device)

def get_cnn_prediction(image):
    print("get cnn pred called")
    input_tensor = transform(image).unsqueeze(0) 
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = cnn_model(input_tensor)          # raw logits
        probs = torch.softmax(outputs, dim=1)      # convert to probabilities
        pred_class = torch.argmax(probs, dim=1).item()
    return class_names[pred_class]