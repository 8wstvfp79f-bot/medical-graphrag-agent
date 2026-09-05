from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.runtime import LocalMedicalRuntime
from src.api.schemas import ChatRequest, ChatResponse
from src.api.service import MedicalChatService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _request_dict(request: ChatRequest) -> dict[str, object]:
    return request.model_dump(exclude_none=True)


def _sse(event: str, data: object) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def create_app(
    service: MedicalChatService | None = None,
    *,
    runtime_factory: Callable[[], LocalMedicalRuntime] = LocalMedicalRuntime,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime: LocalMedicalRuntime | None = None
        if service is not None:
            app.state.chat_service = service
        else:
            runtime = runtime_factory()
            try:
                app.state.chat_service = await asyncio.to_thread(runtime.start)
            except Exception:
                # Startup can fail after one database client has connected. Close
                # anything already opened before FastAPI aborts its lifespan.
                await asyncio.to_thread(runtime.close)
                raise
        try:
            yield
        finally:
            if runtime is not None:
                await asyncio.to_thread(runtime.close)

    application = FastAPI(
        title="Medical GraphRAG Agent API",
        version="0.8.0",
        description=(
            "Evidence-grounded medical retrieval API. It is an educational project "
            "and does not replace professional diagnosis."
        ),
        lifespan=lifespan,
    )
    if FRONTEND_DIR.is_dir():
        application.mount(
            "/assets",
            StaticFiles(directory=FRONTEND_DIR),
            name="frontend-assets",
        )

        @application.get("/", include_in_schema=False)
        async def frontend() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    @application.get("/health")
    async def health(request: Request) -> dict[str, object]:
        return {
            "status": "ok",
            "service_ready": getattr(request.app.state, "chat_service", None)
            is not None,
        }

    @application.post("/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        request: Request,
    ) -> ChatResponse | StreamingResponse:
        chat_service = getattr(request.app.state, "chat_service", None)
        if chat_service is None:
            raise HTTPException(status_code=503, detail="Chat service is not ready")
        request_data = _request_dict(payload)
        if payload.stream:
            async def event_stream() -> AsyncIterator[str]:
                async for item in chat_service.stream_chat(request_data):
                    yield _sse(str(item["event"]), item["data"])

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        try:
            response = await chat_service.chat(request_data)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"error_type": type(exc).__name__, "message": str(exc)},
            ) from exc
        return ChatResponse.model_validate(response)

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
