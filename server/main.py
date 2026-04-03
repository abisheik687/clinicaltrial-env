"""FastAPI application entrypoint."""

from fastapi import FastAPI

from server.api.middleware import install_middleware
from server.api.routes import router


app = FastAPI(title="ClinicalTrialEnv", version="1.0.0")
install_middleware(app)
app.include_router(router)
