import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class_names = ["truck", "convertible", "coupe", "hatchback", "SUV", "sedan", "van", "wagon"]

class MobileNetV2_128(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        self.model = mobilenet_v2(weights=weights)
        
        # First conv layer tweak: MobileNet works fine with 64×64
        # (no need to change kernel/stride like ResNet18, since it’s already lightweight)
        
        # Replace classifier head
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

cnn_model = MobileNetV2_128(num_classes=8)
cnn_model.load_state_dict(torch.load("mobilenetv2_car_classifier.pth", map_location="cpu"))
print("model loaded")
cnn_model.eval()
device = torch.device("cpu")
cnn_model = cnn_model.to(device)
print("before get")

def get_cnn_prediction(image):
    print("get cnn pred called")
    if cnn_model is None:
        return "Model not loaded"
    input_tensor = transform(image.convert('RGB')).unsqueeze(0) 
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = cnn_model(input_tensor)          # raw logits
        probs = torch.softmax(outputs, dim=1)      # convert to probabilities
        pred_class = torch.argmax(probs, dim=1).item()
    return class_names[pred_class]