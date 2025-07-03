from fastapi import FastAPI
from src.controller.api.router import router

app = FastAPI()

app.include_router(router, prefix="/new_rag")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)