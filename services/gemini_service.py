import os
import base64
import json
from typing import Optional

from google import genai


class GeminiService:
    """
    Central Gemini AI service for KarigarKart.

    Handles:
    - AI catalog generation
    - AI product titles
    - AI product descriptions
    - AI image enhancement
    """

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Add it to the .env file."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        # --------------------------------------------------
        # Gemini text models
        #
        # If one model is temporarily unavailable,
        # the service automatically tries the next one.
        # --------------------------------------------------

        self.text_models = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
        ]

        # --------------------------------------------------
        # Gemini image generation / editing model
        # --------------------------------------------------

        self.image_model = "gemini-3.1-flash-image"


    # ======================================================
    # GENERIC TEXT GENERATION
    # ======================================================

    def generate_text(
        self,
        prompt: str
    ) -> str:
        """
        Generate text using Gemini.

        Automatically tries fallback models if a model
        temporarily fails or is unavailable.
        """

        last_error = None

        for model in self.text_models:

            try:

                print(
                    f"🤖 Trying Gemini model: {model}"
                )

                response = (
                    self.client
                    .models
                    .generate_content(
                        model=model,
                        contents=prompt,
                    )
                )

                if response.text:

                    print(
                        f"✅ Gemini response from: {model}"
                    )

                    return response.text.strip()

                last_error = RuntimeError(
                    f"{model} returned an empty response."
                )

                print(
                    f"⚠️ {model} returned empty response"
                )

            except Exception as e:

                last_error = e

                print(
                    f"⚠️ Gemini model {model} failed:"
                )

                print(
                    f"   {str(e)}"
                )

                # Try the next model.
                continue

        raise RuntimeError(
            "All Gemini text models failed. "
            f"Last error: {last_error}"
        )


    # ======================================================
    # AI CATALOG GENERATION
    # ======================================================

    def generate_catalog(
        self,
        description: str,
        category: str = "",
        language: str = "en",
    ) -> dict:
        """
        Generate a structured product catalog entry.

        Returns:

        {
            "title": "...",
            "description": "...",
            "category": "...",
            "tags": [...]
        }
        """

        language_instruction = (
            "Write the output in Hindi."
            if language.lower().startswith("hi")
            else "Write the output in English."
        )

        prompt = f"""
You are an AI catalog assistant for KarigarKart,
an Indian marketplace for handmade artisan products.

Create a professional product listing from the information below.

Product description:
{description}

Category:
{category}

{language_instruction}

Return ONLY valid JSON in exactly this structure:

{{
  "title": "short attractive product title",
  "description": "clear detailed marketplace description",
  "category": "appropriate product category",
  "tags": [
    "tag1",
    "tag2",
    "tag3",
    "tag4",
    "tag5"
  ]
}}

Rules:

- Do not invent materials that are not reasonably supported.
- Do not invent certifications.
- Do not claim the product is handmade unless the input supports it.
- Keep the title concise.
- Make the description suitable for an Indian handicraft marketplace.
- Tags should be useful search keywords.
- Return ONLY JSON.
"""

        raw = self.generate_text(
            prompt
        )

        # --------------------------------------------------
        # Remove accidental Markdown JSON fences.
        # --------------------------------------------------

        cleaned = raw.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]

        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # --------------------------------------------------
        # Parse JSON
        # --------------------------------------------------

        try:

            result = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Gemini returned invalid catalog JSON: "
                + raw
            ) from exc

        return result


    # ======================================================
    # AI PRODUCT TITLE
    # ======================================================

    def generate_product_title(
        self,
        description: str,
        category: str = "",
    ) -> str:
        """
        Generate a marketplace-ready product title.
        """

        prompt = f"""
Create ONE concise and attractive product title
for an Indian handicraft marketplace.

Product description:
{description}

Category:
{category}

Rules:

- Maximum around 80 characters.
- Do not use emojis.
- Do not use quotation marks.
- Do not invent facts.
- Do not add unsupported materials.
- Do not add unsupported certifications.
- Return ONLY the title.
"""

        return self.generate_text(
            prompt
        )


    # ======================================================
    # AI PRODUCT DESCRIPTION
    # ======================================================

    def generate_product_description(
        self,
        description: str,
        category: str = "",
    ) -> str:
        """
        Generate a polished marketplace description.
        """

        prompt = f"""
Write a professional marketplace description for
an Indian artisan or handicraft product.

Product information:
{description}

Category:
{category}

Rules:

- 80-150 words.
- Clear and easy to understand.
- Highlight craftsmanship and product characteristics
  only when supported by the input.
- Do not invent certifications.
- Do not invent materials.
- Do not make unsupported claims.
- Do not exaggerate.
- Return ONLY the description.
"""

        return self.generate_text(
            prompt
        )


    # ======================================================
    # AI IMAGE ENHANCEMENT
    # ======================================================

    def enhance_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: Optional[str] = None,
    ) -> bytes:
        """
        Use Gemini image editing to improve a product
        photograph while preserving the actual product.
        """

        if not prompt:

            prompt = """
Improve this product photograph for an
e-commerce marketplace.

IMPORTANT:
Keep the actual physical product unchanged,
recognizable, and visually consistent.

Improve:

- lighting
- exposure
- sharpness
- clarity
- background cleanliness
- overall product presentation

DO NOT:

- change the product design
- add fake product features
- change the material
- substantially change the product color
- add text
- add logos
- add people
- replace the actual product
- invent product details

Return an improved product photograph.
"""

        print(
            "🖼️ Sending image to Gemini image model..."
        )

        interaction = (
            self.client
            .interactions
            .create(
                model=self.image_model,

                input=[
                    {
                        "type": "image",

                        "data":
                            base64.b64encode(
                                image_bytes
                            ).decode("utf-8"),

                        "mime_type":
                            mime_type,
                    },

                    {
                        "type": "text",

                        "text":
                            prompt,
                    },
                ],
            )
        )

        if not interaction.output_image:

            raise RuntimeError(
                "Gemini did not return an enhanced image."
            )

        print(
            "✅ Gemini image enhancement successful"
        )

        return base64.b64decode(
            interaction.output_image.data
        )