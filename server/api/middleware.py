"""FastAPI middleware and exception handling."""

from fastapi import HTTPException
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from server.utils.logger import configure_logger, log_json


logger = configure_logger()


def install_middleware(app: FastAPI) -> None:
    """Register middleware and exception handlers."""

    @app.middleware("http")
    async def request_logger(request: Request, call_next):
        response = await call_next(request)
        log_json(logger, "request", {"method": request.method, "path": request.url.path, "status_code": response.status_code})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(exc.errors()), "error": "validation_error"},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "error": "http_error"})

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception")
        status = getattr(exc, "status_code", 500)
        detail = getattr(exc, "detail", str(exc))
        return JSONResponse(status_code=status, content={"detail": detail, "error": "server_error"})
