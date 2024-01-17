from prometheus_fastapi_instrumentator import Instrumentator

from fastapi import FastAPI
app = FastAPI()

# your app code here


@app.get("/")
def read_root():
    return {"Hello": "World"}

Instrumentator().instrument(app).expose(app)