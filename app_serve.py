from fastapi import FastAPI
from app.api import bigram_gpt_router
from app.startup import on_startup


def create_app():
    fast_app = FastAPI()
    on_startup(fast_app)
    fast_app.include_router(bigram_gpt_router)

    return fast_app


app = create_app()
