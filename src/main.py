import threading
from pathlib import Path

import uvicorn
import webview
from fastapi import FastAPI
from fastapi.responses import FileResponse

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
app = FastAPI()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND)


if __name__ == "__main__":
    threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "127.0.0.1", "port": 8000},
        daemon=True,
    ).start()
    webview.create_window(
        "Czat testowy",
        "http://127.0.0.1:8000",
        width=1000,
        height=720,
        min_size=(650, 500),
    )
    webview.start()