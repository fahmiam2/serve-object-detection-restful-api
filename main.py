from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.controllers.image_detection import router as image_detection_router
from app.controllers.video_detection import router as video_detection_router

app = FastAPI()

middleware = [
    #TODO: add list of middlewares to prevent cyber attacks here
]

@app.get("/")
def read_root():
    return {"Hello": "World"}

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
