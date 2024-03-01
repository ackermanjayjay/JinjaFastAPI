from typing import Union

from fastapi import FastAPI, Request, Form
from transformers import pipeline
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

pretrained_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
nlp = pipeline("sentiment-analysis", model=pretrained_name, tokenizer=pretrained_name)

templates = Jinja2Templates(directory="templates")

app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="item.html")


@app.get("/classification/{predicts}", response_class=HTMLResponse)
def classifiy(predicts: str, query: str, request: Request):
    if query:
        return templates.TemplateResponse(
            "item.html", {"request": request, "text": query, "result": nlp(query)}
        )
    return templates.TemplateResponse(
        "item.html", {"request": request, "text": "No input detected"}
    )
