from pathlib import Path

from image_enhancer import ImageEnhancer


# Always use the folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent

input_path = BASE_DIR / "test_input.jpg"
output_path = BASE_DIR / "test_output.png"


print("📥 Reading input image...")

with open(input_path, "rb") as file:
    image_bytes = file.read()


enhancer = ImageEnhancer()

print("🤖 Running enhancement...")

result = enhancer.enhance(
    image_bytes
)


with open(output_path, "wb") as file:
    file.write(result)


print("✅ Enhancement completed!")
print(f"📤 Output: {output_path}")