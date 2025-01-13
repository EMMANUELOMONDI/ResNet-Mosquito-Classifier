import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware

from temp import predict


origins = [
    "*",
]

app = FastAPI(title="Mosquito Species Classifier")



@app.post("/classify")
async def extract_speech(audio: UploadFile = File(...)):
    """
    Extract features from an uploaded audio file.
    Returns the processed audio as a WAV file.
    """
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        input_path = os.path.join(temp_dir, "input.wav")
        with open(input_path, "wb") as f:
            f.write(await audio.read())
        output = predict(input_path)

        # Return the processed file
        return Response(
            output, 
            media_type="text/plain"
        )


@app.get("/health")
async def health_check():
    """Check if the service is running."""
    return {"status": "healthy"}


def run():
    import uvicorn

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host="0.0.0.0", port=9000)


if __name__ == "__main__":
    run()