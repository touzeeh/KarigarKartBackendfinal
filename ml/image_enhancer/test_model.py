from pathlib import Path

import torch
from spandrel import ModelLoader, ImageModelDescriptor


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "weights"
    / "RealESRGAN_x4plus.pth"
)


print("📦 Loading Real-ESRGAN model...")
print(f"📍 Model: {MODEL_PATH}")

device = torch.device("cpu")

loader = ModelLoader(device=device)

model = loader.load_from_file(
    str(MODEL_PATH)
)

print("✅ Model loaded successfully!")

print(f"🏗️ Architecture: {model.architecture}")
print(f"🎯 Purpose: {model.purpose}")
print(f"📐 Scale: {model.scale}")
print(f"📥 Input channels: {model.input_channels}")
print(f"📤 Output channels: {model.output_channels}")

assert isinstance(
    model,
    ImageModelDescriptor
)

model.eval()

print("✅ Real-ESRGAN is ready for inference!")