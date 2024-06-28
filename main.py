from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

import whisper
import os

# Load the Whisper model
model = whisper.load_model("base")


#Request body
class AudioFilePath(BaseModel):
    path:str

#initialize the app
app = FastAPI()

#define the endpoint
@app.post("/transcribe/")
def transcribe_audio(file_path:AudioFilePath):
    path=file_path.path
    # Check if the file exists
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Transcribe the audio file
    result = model.transcribe(path)

    # Return the transcription result
    return {"transcription": result["text"]}


 
if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="127.0.0.1", port=8000)
