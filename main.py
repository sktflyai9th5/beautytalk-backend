from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"service": "beautytalk-backend", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}
