# Dependency to provide access to the global variable
from fastapi import Request


def get_gpt_model(request: Request):
    return request.app.state.gpt_bigram_model


def get_gpt_encoder(request: Request):
    return request.app.state.gpt_bigram_encoder


def get_gpt_decoder(request: Request):
    return request.app.state.gpt_bigram_decoder
