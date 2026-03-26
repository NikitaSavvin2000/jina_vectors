import os
from typing import Annotated
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel

from src.config import logger, public_or_local

url = "http://localhost" if public_or_local == "LOCAL" else "http://11.11.11.11"
origins = [url]

app = FastAPI(docs_url="/docs", openapi_url="/jina/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*50)
print(f"        DEVICE     {device} ")
print("="*50)

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v5-text-small", trust_remote_code=True)
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v5-text-small", trust_remote_code=True).to(device)
model = model.float()
model.eval()

BATCH_SIZE = 128

class SentenceItem(BaseModel):
    id: int
    text: str

class BatchRequest(BaseModel):
    batch: list[SentenceItem]

def encode_batch(sentences: list[str], normalize: bool = True):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(outputs, "embeddings"):
            vectors = outputs.embeddings
        elif hasattr(outputs, "last_hidden_state"):
            vectors = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output"):
            vectors = outputs.pooler_output
        else:
            raise AttributeError("Model output has no embeddings / last_hidden_state / pooler_output")
        if normalize:
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
    return vectors.cpu().numpy().tolist()

@app.post("/jina/embeddings")
async def get_embeddings(
        body: Annotated[BatchRequest, Body(example={
            "batch": [
                {"id": 0, "text": "Привет"},
                {"id": 1, "text": "Как дела?"}
            ]
        })],
):

    normalize = True
    try:
        if not body.batch:
            raise HTTPException(status_code=400, detail="No sentences provided")
        ids = [item.id for item in body.batch]
        sentences = [item.text for item in body.batch]

        results = []
        for i in range(0, len(sentences), BATCH_SIZE):
            batch_sentences = sentences[i:i+BATCH_SIZE]
            batch_ids = ids[i:i+BATCH_SIZE]
            vectors = encode_batch(batch_sentences, normalize)
            results.extend([{"id": id_, "vector": v} for id_, v in zip(batch_ids, vectors)])

        return {"predictions": results}
    except Exception as e:
        logger.error(e.__repr__())
        raise HTTPException(status_code=500, detail=f"Unknown error: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Jina V5 embeddings API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7091)