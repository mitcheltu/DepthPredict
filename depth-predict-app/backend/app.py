from flask import Flask, request, send_file
from flask_cors import CORS
import torch
import torchvision.transforms as T
from PIL import Image
import io
import numpy as np
from model import DepthModel  # Assuming model.py is in the same directory
from scipy.ndimage import gaussian_filter, median_filter
import base64
from flask import jsonify
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once at startup
model  = DepthModel().to(device)
model.load_state_dict(torch.load("depth_model.pth", map_location=device))
model.to(device)
model.eval()

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return {"error": "No image file provided"}, 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    original_size = img.size  # (width, height)
    input_tensor = transform(img).unsqueeze(0).to(device)

    print(f"Received image: {file.filename}, shape: {img.size}")

    with torch.no_grad():
        depth = model(input_tensor)[0, 0].cpu().numpy()  # raw depth

    # ----------------------
    # Postprocessing filters
    # ----------------------
    # Gaussian blur
    depth = gaussian_filter(depth, sigma=1)

    # Median filter
    depth = median_filter(depth, size=3)

    # Normalize depth to 0–1 for colormap
    depth_norm_float = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Apply colormap
    colormap = plt.get_cmap('hot')
    depth_colored = (colormap(depth_norm_float)[:, :, :3] * 255).astype(np.uint8)  # RGB 0–255

    # Convert normalized depth to PNG in memory
    depth_img = Image.fromarray(depth_colored).resize(original_size, Image.BILINEAR)

    # Handle Create in-memory PNG file
    buf = io.BytesIO()
    depth_img.save(buf, format="PNG")
    buf.seek(0)

    # Encode PNG to base64
    depth_png_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Return both raw depth and PNG as base64
    return jsonify({
        "depth_png": depth_png_base64,        # can be displayed in browser with data URI
        "depth_raw": depth.tolist()            # actual numeric depth values
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)