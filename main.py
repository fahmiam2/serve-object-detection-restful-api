from app.controllers.image_detection import router as image_detection_router
from app.controllers.video_detection import router as video_detection_router
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import logging

logger = logging.getLogger(__name__)

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
def read_root():
    return {"message": " Hello World"}

"""
if we want to redirect the root into /docs route:

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")
"""

@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}


# Allow CORS (Cross-Origin Resource Sharing)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Include routers/controllers
app.include_router(image_detection_router)
app.include_router(video_detection_router)
