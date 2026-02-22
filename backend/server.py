"""
server.py — FastAPI backend for POV Teleop web UI.

Streams MJPEG video from the pipeline and exposes state/control endpoints.

Run:
    cd backend && python server.py
"""

import asyncio
import signal
import threading

import janus
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from pipeline import FrameBundle, run_pipeline

# Shared state — set up before either thread starts
_queue: janus.Queue | None = None
_stop_event = threading.Event()

# Latest frame bundle (updated by consumer task)
_latest: FrameBundle | None = None
_latest_lock = asyncio.Lock()


async def _consume_frames():
    """Async task that reads from janus async queue and updates _latest."""
    global _latest
    if _queue is None:
        return
    async_q = _queue.async_q
    while not _stop_event.is_set():
        try:
            bundle = await asyncio.wait_for(async_q.get(), timeout=0.5)
            async with _latest_lock:
                _latest = bundle
        except asyncio.TimeoutError:
            continue
        except Exception:
            break


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the frame consumer. Pipeline runs on main thread, not here."""
    consumer = asyncio.create_task(_consume_frames())
    yield
    _stop_event.set()
    consumer.cancel()


app = FastAPI(title="POV Teleop", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── MJPEG Streams ──────────────────────────────────────────


async def _mjpeg_generator(frame_attr: str):
    """Yield MJPEG multipart frames from the latest bundle."""
    boundary = b"--frame\r\n"
    while not _stop_event.is_set():
        async with _latest_lock:
            bundle = _latest
        if bundle is not None:
            jpeg = getattr(bundle, frame_attr, None)
            if jpeg:
                yield (
                    boundary
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
        await asyncio.sleep(1 / 30)


@app.get("/api/stream/sim")
async def stream_sim():
    return StreamingResponse(
        _mjpeg_generator("sim_jpeg"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/camera")
async def stream_camera():
    return StreamingResponse(
        _mjpeg_generator("camera_jpeg"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── State ──────────────────────────────────────────


@app.get("/api/state")
async def get_state():
    async with _latest_lock:
        bundle = _latest
    if bundle is None:
        return JSONResponse(
            {"right_action": [], "left_action": [], "fps": 0, "connected": False}
        )
    return JSONResponse(
        {
            "right_action": bundle.right_action,
            "left_action": bundle.left_action,
            "fps": bundle.fps,
            "connected": True,
        }
    )


# ── WebSocket (iOS fallback) ──────────────────────────────


@app.websocket("/api/ws")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    try:
        while not _stop_event.is_set():
            async with _latest_lock:
                bundle = _latest
            if bundle is not None and bundle.sim_jpeg:
                await ws.send_bytes(bundle.sim_jpeg)
            await asyncio.sleep(1 / 30)
    except WebSocketDisconnect:
        pass


# ── Control ──────────────────────────────────────────


@app.post("/api/stop")
async def stop():
    _stop_event.set()
    return {"status": "stopping"}


# ── Main: pipeline on main thread, uvicorn in background ────


if __name__ == "__main__":
    _queue = janus.Queue(maxsize=2)

    # Start uvicorn in a daemon thread (async IO doesn't need main thread)
    server_config = uvicorn.Config(
        app, host="0.0.0.0", port=8000, log_level="info"
    )
    server = uvicorn.Server(server_config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Run pipeline on main thread (MuJoCo/GLFW needs main thread on macOS)
    def _handle_sigint(sig, frame):
        print("\nShutting down...")
        _stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        run_pipeline(_queue, _stop_event)
    finally:
        _stop_event.set()
        server.should_exit = True
        server_thread.join(timeout=5)
        _queue.close()
