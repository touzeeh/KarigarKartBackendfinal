import base64
import io
import json
import math
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO

from services.gemini_service import GeminiService
from ml.image_enhancer.image_enhancer import ImageEnhancer


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).parent
ML = ROOT / "ml"


# ============================================================
# GLOBAL STATE
# ============================================================

state = {}


# ============================================================
# STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # ----------------------------
    # Load ML models
    # ----------------------------

    try:
        state["detector"] = YOLO(
            str(ML / "new_ml" / "best.pt")
        )

        state["metadata"] = json.loads(
            (ML / "new_ml" / "class_metadata.json").read_text()
        )

        expected = {
            int(k): v["class_name"]
            for k, v in state["metadata"].items()
        }

        if state["detector"].names != expected:
            raise ValueError("Detector taxonomy mismatch")

        state["price"] = joblib.load(
            ML / "price_ml" / "models" / "price_pipeline.joblib"
        )

        state["price_info"] = json.loads(
            (ML / "price_ml" / "models" / "model_info.json").read_text()
        )

        state["ready"] = True

        print("✅ ML models loaded successfully")

    except Exception as e:

        state["ready"] = False
        state["error"] = str(e)

        print("❌ ML model loading failed:", e)


    # ----------------------------
    # Initialize Gemini
    # ----------------------------

    try:

        state["gemini"] = GeminiService()

        state["gemini_ready"] = True

        print("✅ Gemini AI initialized successfully")

    except Exception as e:

        state["gemini"] = None
        state["gemini_ready"] = False
        state["gemini_error"] = str(e)

        print("⚠️ Gemini AI unavailable:", e)


    # ----------------------------
    # Initialize local image enhancer
    # ----------------------------

    try:

        state["image_enhancer"] = ImageEnhancer()
        state["image_enhancer_ready"] = True

        print("✅ Local ML image enhancer initialized")

    except Exception as e:

        state["image_enhancer"] = None
        state["image_enhancer_ready"] = False
        state["image_enhancer_error"] = str(e)

        print("⚠️ Local ML image enhancer unavailable:", e)


    yield


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="KarigarKart Fresh Backend",
    lifespan=lifespan,
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
@app.get("/api/v1/health")
def health():

    return {
        "status": (
            "ok"
            if state.get("ready")
            else "degraded"
        ),

        "models_loaded": bool(
            state.get("ready")
        ),

        "gemini_ready": bool(
            state.get("gemini_ready")
        ),

        "image_enhancer_ready": bool(
            state.get("image_enhancer_ready")
        ),

        "error": state.get("error"),

        "gemini_error": state.get("gemini_error"),

        "image_enhancer_error": state.get(
            "image_enhancer_error"
        ),
    }


# ============================================================
# HELPERS
# ============================================================

def ready():

    if not state.get("ready"):

        raise HTTPException(
            503,
            "ML models unavailable: "
            + str(state.get("error", "not loaded"))
        )


def gemini_ready():

    if not state.get("gemini_ready"):

        raise HTTPException(
            503,
            "Gemini AI unavailable: "
            + str(
                state.get(
                    "gemini_error",
                    "Gemini not configured"
                )
            )
        )


def decode_image(raw):

    if len(raw) > 10_000_000:

        raise HTTPException(
            413,
            "Image exceeds 10 MB"
        )

    try:

        im = Image.open(
            io.BytesIO(raw)
        )

        im.verify()

        im = Image.open(
            io.BytesIO(raw)
        )

        im = ImageOps.exif_transpose(
            im
        ).convert("RGB")

        if im.width * im.height > 20_000_000:

            raise HTTPException(
                413,
                "Image resolution too large"
            )

        return im

    except HTTPException:
        raise

    except Exception:

        raise HTTPException(
            415,
            "Invalid image"
        )


# ============================================================
# ML DETECTION
# ============================================================

def detect(im):

    ready()

    result = state["detector"].predict(
        np.asarray(im),
        conf=0.4,
        imgsz=640,
        device="cpu",
        verbose=False,
    )[0]

    objects = []

    if result.masks is not None:

        for box, poly in zip(
            result.boxes,
            result.masks.xyn
        ):

            cls = int(
                box.cls.item()
            )

            info = state["metadata"][
                str(cls)
            ]

            objects.append({

                "object_id":
                    len(objects) + 1,

                "class_name":
                    info["class_name"],

                "handicraft_type":
                    info["handicraft_type"],

                "material":
                    info["material"],

                "category_family":
                    info.get("category_family"),

                "joint_confidence":
                    float(
                        box.conf.item()
                    ),

                "object_mask_polygon_normalized":
                    poly.tolist(),

                "bounding_box_xyxy_pixels":
                    box.xyxy[0].tolist(),

                "material_status":
                    "visual_estimate_not_authenticated",
            })

    return {

        "image_width_px":
            im.width,

        "image_height_px":
            im.height,

        "detected_object_count":
            len(objects),

        "objects":
            objects,
    }


# ============================================================
# ML PRICE
# ============================================================

def price(class_name, product_form):

    ready()

    info = state["price_info"]

    if class_name not in info["training_class_counts"]:

        raise HTTPException(
            422,
            "Class not present in price-model training data"
        )

    if product_form not in info["training_forms"]:

        raise HTTPException(
            422,
            "Unknown product_form; choose one of "
            + ", ".join(info["training_forms"])
        )

    model = state["price"]

    raw = float(
        model.predict(
            pd.DataFrame([
                {
                    "class_name": class_name,
                    "product_form": product_form,
                }
            ])
        )[0]
    )

    if not math.isfinite(raw):

        raise HTTPException(
            503,
            "Invalid model output"
        )

    output = os.getenv(
        "PRICE_PIPELINE_OUTPUT",
        "log1p"
    )

    if output == "log1p":

        value = math.expm1(raw)

    elif output == "inr":

        value = raw

    else:

        raise HTTPException(
            503,
            "Invalid PRICE_PIPELINE_OUTPUT configuration"
        )

    return {

        "listed_price_estimate_inr":
            round(max(0, value), 2),

        "class_name":
            class_name,

        "product_form":
            product_form,

        "basis":
            "catalogue listing estimate, not verified market price",

        "training_class_products":
            info["training_class_counts"][class_name],
    }


# ============================================================
# ML PRICE REQUEST
# ============================================================

class PriceRequest(BaseModel):

    class_name: str

    product_form: str = "unspecified"


@app.post("/api/v1/ml/price")
def predict_price(req: PriceRequest):

    return price(
        req.class_name,
        req.product_form
    )


# ============================================================
# ML IMAGE DETECTION
# ============================================================

@app.post("/api/v1/ml/detect")
async def predict_image(
    image: UploadFile = File(...)
):

    raw = await image.read()

    return detect(
        decode_image(raw)
    )


# ============================================================
# COMBINED PRODUCT ANALYSIS
# ============================================================

@app.post("/api/v1/product/analyze")
async def analyze(
    image: UploadFile = File(...),
    product_form: str = Form("unspecified"),
    object_index: int | None = Form(None),
):

    raw = await image.read()

    result = detect(
        decode_image(raw)
    )

    objs = result["objects"]

    if not objs:

        raise HTTPException(
            422,
            "No product detected"
        )

    if (
        len(objs) > 1
        and object_index is None
    ):

        return {

            "selection_required": True,

            "detection": result,

            "message":
                "Select an object_index before pricing",
        }

    idx = (
        object_index
        if object_index is not None
        else 0
    )

    if idx < 0 or idx >= len(objs):

        raise HTTPException(
            422,
            "Invalid object_index"
        )

    return {

        "selection_required": False,

        "detection": result,

        "selected_object_index": idx,

        "price": price(
            objs[idx]["class_name"],
            product_form
        ),
    }


# ============================================================
# AI PRICING
# ============================================================

class PricingRequest(BaseModel):

    description: str = ""

    category: str = ""

    image_base64: str | None = None

    product_form: str = "unspecified"

    raw_material_cost: float = Field(
        0,
        ge=0
    )

    labor_cost: float = Field(
        0,
        ge=0
    )

    electricity_cost: float = Field(
        0,
        ge=0
    )

    packaging_cost: float = Field(
        0,
        ge=0
    )

    transport_cost: float = Field(
        0,
        ge=0
    )

    platform_fee: float = Field(
        0,
        ge=0
    )

    other_cost: float = Field(
        0,
        ge=0
    )

    desired_margin_percent: float = Field(
        20,
        ge=0,
        le=1000
    )


@app.post("/ai/pricing")
def pricing(req: PricingRequest):

    costs = sum([
        req.raw_material_cost,
        req.labor_cost,
        req.electricity_cost,
        req.packaging_cost,
        req.transport_cost,
        req.platform_fee,
        req.other_cost,
    ])

    floor = math.ceil(
        costs * (
            1 + req.desired_margin_percent / 100
        )
    )

    result = None
    warning = None

    if req.image_base64:

        try:

            raw = base64.b64decode(
                req.image_base64.split(",")[-1],
                validate=True
            )

            objects = detect(
                decode_image(raw)
            )["objects"]

            if len(objects) == 1:

                result = price(
                    objects[0]["class_name"],
                    req.product_form
                )

            elif len(objects) > 1:

                warning = (
                    "Multiple products detected; "
                    "select a single product for an ML estimate"
                )

            else:

                warning = (
                    "No supported handicraft detected"
                )

        except (
            ValueError,
            HTTPException
        ) as e:

            warning = str(e)

    suggested = (
        max(
            floor,
            round(
                result[
                    "listed_price_estimate_inr"
                ]
            )
        )
        if result
        else floor
    )

    return {

        "success": True,

        "suggested_price":
            suggested,

        "price_range": {

            "low":
                floor,

            "high":
                suggested,
        },

        "b2b_price":
            floor,

        "b2c_price":
            suggested,

        "cost_floor":
            floor,

        "ml_estimate":
            result,

        "market_basis":
            (
                "Historical catalogue listings"
                if result
                else "Cost-based calculation only"
            ),

        "reasoning":
            (
                "Cost floor compared with a model "
                "estimate of catalogue listing price."
                if result
                else
                "No ML estimate available; "
                "used supplied costs and markup."
            ),

        "warning":
            warning,
    }


# ============================================================
# GEMINI CATALOG GENERATION
# ============================================================

@app.post("/ai/catalog")
@app.post("/ai/catalog/generate")
def catalog(payload: dict):

    gemini_ready()

    # ----------------------------------------
    # Accept several possible frontend keys
    # ----------------------------------------

    description = (
        payload.get("description")
        or payload.get("product_description")
        or payload.get("text")
        or payload.get("transcript")
        or ""
    )

    category = (
        payload.get("category")
        or payload.get("category_family")
        or ""
    )

    language = (
        payload.get("language")
        or "en"
    )

    # ----------------------------------------
    # If frontend sends an image, include
    # basic detected information when possible.
    # ----------------------------------------

    image_base64 = (
        payload.get("image_base64")
        or payload.get("image")
    )

    detected_context = ""

    if image_base64:

        try:

            raw = base64.b64decode(
                image_base64.split(",")[-1],
                validate=True
            )

            detected = detect(
                decode_image(raw)
            )

            if detected["objects"]:

                detected_context = json.dumps(
                    detected["objects"],
                    ensure_ascii=False
                )

        except Exception as e:

            print(
                "⚠️ Optional image detection failed:",
                e
            )

    if not description:

        description = (
            "Handmade artisan product"
        )

    if detected_context:

        description += (
            "\n\nVisual ML detection context:\n"
            + detected_context
        )

    try:

        result = state["gemini"].generate_catalog(
            description=description,
            category=category,
            language=language,
        )

    except Exception as e:

        print(
            "❌ Gemini catalog generation failed:",
            e
        )

        raise HTTPException(
            502,
            "Gemini catalog generation failed: "
            + str(e)
        )

    # ----------------------------------------
    # Return both direct fields and a draft
    # ----------------------------------------
    # This makes the endpoint easier to consume
    # from the existing Flutter code.
    # ----------------------------------------

    return {

        "success": True,

        "title":
            result.get("title", ""),

        "description":
            result.get("description", ""),

        "category":
            result.get("category", category),

        "tags":
            result.get("tags", []),

        "draft":
            result,
    }


# ============================================================
# AI PRODUCT TITLE
# ============================================================

@app.post("/ai/catalog/title")
def generate_title(payload: dict):

    gemini_ready()

    description = (
        payload.get("description")
        or payload.get("product_description")
        or payload.get("text")
        or ""
    )

    category = (
        payload.get("category")
        or ""
    )

    if not description:

        raise HTTPException(
            422,
            "description is required"
        )

    try:

        title = state["gemini"].generate_product_title(
            description=description,
            category=category,
        )

    except Exception as e:

        print(
            "❌ Gemini title generation failed:",
            e
        )

        raise HTTPException(
            502,
            "Gemini title generation failed: "
            + str(e)
        )

    return {

        "success": True,

        "title":
            title,
    }


# ============================================================
# AI PRODUCT DESCRIPTION
# ============================================================

@app.post("/ai/catalog/description")
def generate_description(payload: dict):

    gemini_ready()

    description = (
        payload.get("description")
        or payload.get("product_description")
        or payload.get("text")
        or ""
    )

    category = (
        payload.get("category")
        or ""
    )

    if not description:

        raise HTTPException(
            422,
            "description is required"
        )

    try:

        generated = (
            state["gemini"]
            .generate_product_description(
                description=description,
                category=category,
            )
        )

    except Exception as e:

        print(
            "❌ Gemini description generation failed:",
            e
        )

        raise HTTPException(
            502,
            "Gemini description generation failed: "
            + str(e)
        )

    return {

        "success": True,

        "description":
            generated,
    }


# ============================================================
# AI TRANSCRIPTION
# ============================================================

@app.post("/ai/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = "hi",
):

    gemini_ready()

    audio_bytes = await file.read()

    if not audio_bytes:

        raise HTTPException(
            400,
            "Empty audio file"
        )

    if len(audio_bytes) > 25_000_000:

        raise HTTPException(
            413,
            "Audio file is too large"
        )

    mime_type = (
        file.content_type
        or "audio/wav"
    )

    temp_path = None

    try:

        suffix = Path(
            file.filename or "audio.wav"
        ).suffix

        if not suffix:

            suffix = ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            temp.write(audio_bytes)

            temp_path = temp.name

        # Upload the audio through the Gemini Files API.
        audio_file = (
            state["gemini"]
            .client
            .files
            .upload(
                file=temp_path
            )
        )

        interaction = (
            state["gemini"]
            .client
            .interactions
            .create(
                model="gemini-3.5-transcribe",

                input=[
                    {
                        "type": "audio",

                        "uri":
                            audio_file.uri,

                        "mime_type":
                            audio_file.mime_type
                            or mime_type,
                    }
                ],

                generation_config={
                    "transcription_config": {

                        "language_codes": (
                            [language]
                            if language
                            else []
                        ),

                        "mode": "smart",
                    }
                },
            )
        )

        transcript = (
            interaction.output_text
            or ""
        ).strip()

        if not transcript:

            raise RuntimeError(
                "Gemini returned an empty transcript."
            )

        return {

            "success": True,

            "transcript":
                transcript,

            "text":
                transcript,

            "language":
                language,
        }

    except HTTPException:

        raise

    except Exception as e:

        print(
            "❌ Gemini transcription failed:",
            e
        )

        raise HTTPException(
            502,
            "Gemini transcription failed: "
            + str(e)
        )

    finally:

        if temp_path:

            try:

                os.remove(temp_path)

            except OSError:

                pass


# ============================================================
# AI IMAGE ENHANCEMENT
# ============================================================

@app.post("/ai/enhance")
@app.post("/ai/studio/enhance")
async def studio_enhance(
    image: UploadFile = File(...)
):
    """
    Local ML image enhancement.

    Pipeline:
        Upload
          ↓
        Image validation / EXIF correction
          ↓
        Safe resize for inference
          ↓
        OpenCV enhancement
          ↓
        Real-ESRGAN neural super-resolution
          ↓
        PNG response

    Gemini is NOT used by this endpoint.
    """

    # ---------------------------------------------------------
    # Check local ML enhancer
    # ---------------------------------------------------------

    if not state.get("image_enhancer_ready"):
        raise HTTPException(
            503,
            "Local ML image enhancer unavailable: "
            + str(
                state.get(
                    "image_enhancer_error",
                    "model not loaded"
                )
            )
        )

    # ---------------------------------------------------------
    # Read uploaded image
    # ---------------------------------------------------------

    raw = await image.read()

    if not raw:
        raise HTTPException(
            400,
            "Empty image file"
        )

    # ---------------------------------------------------------
    # Validate and decode image
    # ---------------------------------------------------------

    im = decode_image(raw)

    # ---------------------------------------------------------
    # Protect CPU/RAM from huge Real-ESRGAN requests
    #
    # Real-ESRGAN is an x4 model. We cap the longest input side
    # at 1200 px, so the maximum output side is about 4800 px.
    # ---------------------------------------------------------

    max_side = 1200

    if max(im.width, im.height) > max_side:

        scale = max_side / max(
            im.width,
            im.height
        )

        new_width = max(
            1,
            int(im.width * scale)
        )

        new_height = max(
            1,
            int(im.height * scale)
        )

        print(
            f"📉 Resizing large enhancement input: "
            f"{im.width}x{im.height} -> "
            f"{new_width}x{new_height}"
        )

        im = im.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    # ---------------------------------------------------------
    # Re-encode to JPEG so the enhancer receives a normalized
    # RGB image regardless of the original upload format.
    # ---------------------------------------------------------

    normalized = io.BytesIO()

    im.save(
        normalized,
        format="JPEG",
        quality=95
    )

    normalized_raw = normalized.getvalue()

    print(
        f"🖼️ Enhancing image: "
        f"{image.filename or 'uploaded_image'}"
    )

    print(
        f"📐 Enhancement input: "
        f"{im.width}x{im.height}"
    )

    # ---------------------------------------------------------
    # Run CPU-heavy inference outside the async event loop.
    # The Real-ESRGAN model was loaded once at startup.
    # ---------------------------------------------------------

    try:

        enhanced_bytes = await run_in_threadpool(
            state["image_enhancer"].enhance,
            normalized_raw
        )

    except Exception as e:

        print(
            "❌ Local ML image enhancement failed:",
            e
        )

        raise HTTPException(
            500,
            "Local ML image enhancement failed: "
            + str(e)
        )

    # ---------------------------------------------------------
    # Return enhanced image
    # ---------------------------------------------------------

    return Response(
        content=enhanced_bytes,
        media_type="image/png",

        headers={
            "X-AI-Provider":
                "KarigarKart Local ML",

            "X-AI-Model":
                "Real-ESRGAN x4plus",

            "X-AI-Feature":
                "product-image-enhancement",
        },
    )
