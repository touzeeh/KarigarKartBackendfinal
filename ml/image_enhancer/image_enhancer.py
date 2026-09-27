from pathlib import Path

import cv2
import numpy as np
import torch
from spandrel import ModelLoader


# Keep CPU inference reasonably fast without trying to use every logical
# thread on the machine.
_CPU_THREADS = min(8, max(1, torch.get_num_threads()))
torch.set_num_threads(_CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch only allows this to be changed before inter-op work starts.
    pass


class ImageEnhancer:
    """
    KarigarKart local ML image-enhancement pipeline.

    Pipeline:
        uploaded image
            ↓
        safe input resize
            ↓
        OpenCV quality enhancement
            ↓
        Real-ESRGAN x4plus
            ↓
        PNG bytes

    Gemini is not used here.
    """

    # Processing a smaller image makes CPU Real-ESRGAN much faster.
    # The final image is still enlarged 4x by Real-ESRGAN.
    MAX_INPUT_SIDE = 320

    def __init__(self):
        print("🖼️ Initializing KarigarKart Image Enhancer...")
        print(f"🧵 PyTorch CPU threads: {_CPU_THREADS}")

        self.device = torch.device("cpu")

        model_path = (
            Path(__file__).resolve().parent
            / "weights"
            / "RealESRGAN_x4plus.pth"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Real-ESRGAN model not found: {model_path}"
            )

        print("📦 Loading Real-ESRGAN...")

        loader = ModelLoader(device=self.device)

        self.model = loader.load_from_file(
            str(model_path)
        )

        self.model.eval()

        print("✅ Real-ESRGAN loaded")
        print(f"📐 Scale: {self.model.scale}")

    def _resize_for_inference(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Limit the longest input side before neural inference.
        This prevents large uploads from causing very slow CPU
        inference or excessive RAM usage.
        """

        height, width = image.shape[:2]
        longest_side = max(height, width)

        if longest_side <= self.MAX_INPUT_SIDE:
            return image

        scale = self.MAX_INPUT_SIDE / longest_side

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        print(
            f"📉 ML input resize: "
            f"{width}x{height} -> "
            f"{new_width}x{new_height}"
        )

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    def basic_enhancement(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Fast OpenCV enhancement before neural restoration.
        """

        # ---------------------------------------------------------
        # Local contrast / exposure
        # ---------------------------------------------------------

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l_channel = clahe.apply(l_channel)

        lab = cv2.merge(
            (
                l_channel,
                a_channel,
                b_channel
            )
        )

        enhanced = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

        # ---------------------------------------------------------
        # Mild denoising
        # ---------------------------------------------------------

        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            3,
            3,
            7,
            21
        )

        # ---------------------------------------------------------
        # Controlled sharpening
        # ---------------------------------------------------------

        blurred = cv2.GaussianBlur(
            enhanced,
            (0, 0),
            1.2
        )

        enhanced = cv2.addWeighted(
            enhanced,
            1.20,
            blurred,
            -0.20,
            0
        )

        return np.clip(
            enhanced,
            0,
            255
        ).astype(np.uint8)

    def neural_enhancement(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Run Real-ESRGAN on CPU.
        """

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        tensor = (
            torch.from_numpy(rgb)
            .float()
            .div(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )

        tensor = tensor.to(self.device)

        print("🤖 Running Real-ESRGAN inference...")

        with torch.inference_mode():
            output = self.model(tensor)

        output = output.squeeze(0)
        output = output.clamp(0, 1)

        output = (
            output
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        output = (
            output * 255.0
        ).round().astype(np.uint8)

        output = cv2.cvtColor(
            output,
            cv2.COLOR_RGB2BGR
        )

        return output

    def enhance(
        self,
        image_bytes: bytes
    ) -> bytes:
        """
        Accept image bytes and return enhanced PNG bytes.
        """

        print("📥 Decoding image...")

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                "Could not decode uploaded image."
            )

        original_height, original_width = image.shape[:2]

        print(
            f"📐 Original input: "
            f"{original_width}x{original_height}"
        )

        # ---------------------------------------------------------
        # 1. Keep CPU inference bounded
        # ---------------------------------------------------------

        image = self._resize_for_inference(
            image
        )

        # ---------------------------------------------------------
        # 2. Fast visual enhancement
        # ---------------------------------------------------------

        print("✨ Applying basic image enhancement...")

        enhanced = self.basic_enhancement(
            image
        )

        # ---------------------------------------------------------
        # 3. Neural super-resolution
        # ---------------------------------------------------------

        enhanced = self.neural_enhancement(
            enhanced
        )

        output_height, output_width = enhanced.shape[:2]

        print(
            f"📐 Neural output: "
            f"{output_width}x{output_height}"
        )

        # ---------------------------------------------------------
        # 4. Encode as PNG
        # ---------------------------------------------------------

        success, encoded = cv2.imencode(
            ".png",
            enhanced,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                3
            ]
        )

        if not success:
            raise RuntimeError(
                "Failed to encode enhanced image."
            )

        print("✅ Image enhancement completed!")

        return encoded.tobytes()
