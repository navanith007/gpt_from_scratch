from fastapi import FastAPI
from app.api import bigram_router, bigram_gpt_router


def create_app():
    fast_app = FastAPI()
    fast_app.include_router(bigram_router)
    fast_app.include_router(bigram_gpt_router)
    return fast_app


app = create_app()
