from pathlib import Path
import gc

import cv2
import numpy as np
import torch
from spandrel import ModelLoader


# Keep CPU inference reasonably fast without trying to use every logical
# thread on the machine.
_CPU_THREADS = 1
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
    MAX_INPUT_SIDE = 256
    TILE_SIZE = 64
    TILE_PAD = 8

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

    def _run_esrgan_tile(self, tile: np.ndarray) -> np.ndarray:
        """Run one small RGB tile through Real-ESRGAN."""
        rgb = np.ascontiguousarray(tile)

        tensor = (
            torch.from_numpy(rgb)
            .float()
            .div_(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )

        print(
            f"🧩 ESRGAN tile inference: "
            f"{rgb.shape[1]}x{rgb.shape[0]}"
        )

        with torch.inference_mode():
            output = self.model(tensor)

        output = output.squeeze(0).clamp_(0, 1)

        output = (
            output
            .permute(1, 2, 0)
            .mul_(255.0)
            .round_()
            .byte()
            .cpu()
            .numpy()
        )

        result = np.ascontiguousarray(output)

        del output
        del tensor
        gc.collect()

        return result

    def neural_enhancement(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Run Real-ESRGAN using low-memory tiled inference.

        The previous full-frame forward pass could exceed Railway's
        1 GB replica memory limit. Tiles keep peak activation memory low.
        """

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        height, width = rgb.shape[:2]
        scale = int(getattr(self.model, "scale", 4) or 4)

        output = np.empty(
            (
                height * scale,
                width * scale,
                3
            ),
            dtype=np.uint8
        )

        tile_size = self.TILE_SIZE
        tile_pad = self.TILE_PAD

        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        total_tiles = tiles_x * tiles_y
        tile_number = 0

        print(
            "🤖 Running tiled Real-ESRGAN inference "
            f"(tile={tile_size}, pad={tile_pad}, "
            f"tiles={total_tiles})..."
        )

        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                tile_number += 1

                x1 = min(x0 + tile_size, width)
                y1 = min(y0 + tile_size, height)

                # Overlap the input tiles to reduce boundary artifacts.
                px0 = max(0, x0 - tile_pad)
                py0 = max(0, y0 - tile_pad)
                px1 = min(width, x1 + tile_pad)
                py1 = min(height, y1 + tile_pad)

                tile = rgb[py0:py1, px0:px1]

                print(
                    f"🧩 Tile {tile_number}/{total_tiles}: "
                    f"{tile.shape[1]}x{tile.shape[0]}"
                )

                tile_output = self._run_esrgan_tile(tile)

                # Remove the padded border from the neural output.
                crop_left = (x0 - px0) * scale
                crop_top = (y0 - py0) * scale
                crop_right = crop_left + (x1 - x0) * scale
                crop_bottom = crop_top + (y1 - y0) * scale

                clean = tile_output[
                    crop_top:crop_bottom,
                    crop_left:crop_right
                ]

                ox0 = x0 * scale
                oy0 = y0 * scale
                ox1 = x1 * scale
                oy1 = y1 * scale

                expected_h = oy1 - oy0
                expected_w = ox1 - ox0

                if clean.shape[:2] != (expected_h, expected_w):
                    raise RuntimeError(
                        "Unexpected Real-ESRGAN tile output shape: "
                        f"{clean.shape[:2]} != "
                        f"{(expected_h, expected_w)}"
                    )

                output[oy0:oy1, ox0:ox1] = clean

                del clean
                del tile_output
                del tile
                gc.collect()

        del rgb
        gc.collect()

        # Convert RGB output back to the BGR convention used by the
        # existing OpenCV encoder.
        return cv2.cvtColor(
            output,
            cv2.COLOR_RGB2BGR
        )

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
