import sys
print(sys.executable)
print(sys.version)

import os
import torch
from model import DepthModel
import onnx
print(onnx.__version__)

# Get path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "depth_model.pth")

# Choose device
device = torch.device("cpu")  # free Render tier has no GPU

# Load the model
model = DepthModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("Model loaded from:", model_path)

dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(model, dummy_input, "depth_model.onnx", opset_version=12)