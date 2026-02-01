import os
import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse
from fastapi import Response
from src.textSummarizer.pipeline.prediction_pipeline import PredictionPipeline


text: str = "What is text summarization ?"

app = FastAPI()


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training Successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict(text):
    try:
        pipeline = PredictionPipeline()
        summarized_text = pipeline.predict(text)
        return Response(f"Summarized Text: {summarized_text}")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
