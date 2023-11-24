from fastapi import FastAPI
from app.api.bigram_serve import bigram_router


def create_app():
    fast_app = FastAPI()
    fast_app.include_router(bigram_router)
    return fast_app


app = create_app()
