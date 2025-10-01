from fastapi import FastAPI
import uvicorn
from sentiment_api import router as sentiment_router
from cv_api import router as cv_router

app = FastAPI(title="Unified AI API", version="1.0")
app.include_router(sentiment_router)
app.include_router(cv_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)