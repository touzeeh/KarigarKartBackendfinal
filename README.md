# KarigarKart fresh backend starter

## Install models

Copy the following exact files from https://github.com/anshkumar-ops/karigarkart-ml:

- `new_ml/best.pt` -> `ml/new_ml/best.pt`
- `new_ml/class_metadata.json` -> `ml/new_ml/class_metadata.json`
- `price_ml/models/price_pipeline.joblib` -> `ml/price_ml/models/price_pipeline.joblib`
- `price_ml/models/model_info.json` -> `ml/price_ml/models/model_info.json`

## Local run

Use Python 3.13. Create a virtual environment, install `requirements.txt`, then run:

`uvicorn main:app --reload`

Visit `http://localhost:8000/health` and `http://localhost:8000/docs`.

## Railway

Push this folder (including the model files) to a NEW GitHub repo. Railway -> New Project -> Deploy from GitHub -> select repo. Dockerfile auto-detected. Generate public domain. Set `PRICE_PIPELINE_OUTPUT` after checking the serialized pipeline's output behavior. A healthy HTTP response alone does not establish that the pricing numbers are correct.

## Current limitations

- Catalog generation, transcription and studio enhancement return 503 until real AI providers are connected. They are NOT implemented by the two supplied ML models.
- `/ai/pricing` supports the existing Flutter cost fields and optionally uses image detection + price model. The price model uses `class_name` and `product_form` only; it does not consume costs or image size. If multiple objects are detected, the endpoint uses cost-only pricing rather than arbitrarily selecting one.
- The model was evaluated against catalogue listing prices, not live market data or real-world sales. No guarantee of a fair selling price.
- A single Railway container with Ultralytics/PyTorch may need more memory than an available free allowance. Confirm current plan and actual deployment memory.
- Before exposing publicly, add authentication, rate limits, request quotas, CORS allowlist, and observability.
