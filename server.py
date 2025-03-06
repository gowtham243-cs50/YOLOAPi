# main.py (or server.py)
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "active", "message": "API is running"}
