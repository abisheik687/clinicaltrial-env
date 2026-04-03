"""OpenEnv validator shim for the FastAPI application."""

import uvicorn

from server.main import app


def main() -> None:
    """Console entry point for OpenEnv packaging validation."""
    uvicorn.run("server.main:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
