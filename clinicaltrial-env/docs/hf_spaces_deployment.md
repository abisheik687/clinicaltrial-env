# Hugging Face Spaces Deployment Checklist

This document is the final handoff checklist for deploying ClinicalTrialEnv as a Docker Space.

## 1. Create the Space

- Go to Hugging Face Spaces
- Create a new Space named `clinicaltrial-env`
- Choose `Docker` as the SDK
- Set visibility as needed

## 2. Push This Project

Push the full repository contents, including:

- `Dockerfile`
- `openenv.yaml`
- `inference.py`
- `server/`
- `protocols/`
- `README.md`

## 3. Configure Space Variables

Add the following secrets or variables in the Space settings:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:

- `ENV_URL`
- `LOG_LEVEL`
- `SESSION_TIMEOUT_MINUTES`

## 4. Confirm Required Space Metadata

The README front matter already includes:

- `sdk: docker`
- `app_port: 7860`
- healthcare and OpenEnv tags

## 5. Wait for Build Completion

After push:

- confirm the Docker build succeeds
- open the Space homepage
- confirm `GET /health` works
- confirm `POST /reset` returns `200`

## 6. Final Smoke Tests

Example checks:

```bash
curl https://YOUR_SPACE_URL/health
curl -X POST https://YOUR_SPACE_URL/reset -H "Content-Type: application/json" -d '{}'
```

## 7. Submission-Day Notes

- keep the Space awake before running your final checks
- do not rename `inference.py`
- do not remove `openenv.yaml`
- do not change the API route names
- do not change the log format in `inference.py`

