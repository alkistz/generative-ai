from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, field_validator

from models import generate_audio, generate_text, load_audio_model, load_text_model
from schemas import VoicePresets
from utils import audio_array_to_buffer


class UserCreate(BaseModel):
    username: str
    password: str

    @field_validator("password")
    def validate_password(cls, value):
        if len(value) < 8:
            raise ValueError("Password must be at least 8 charactes long")
        return value


app = FastAPI()
openai_client = OpenAI(api_key="some key")


@app.get(
    "/generate/audio",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
)
def serve_text_to_audio_model_controller(
    prompt: str, preset: VoicePresets = "v2/en_speaker_1"
):
    processor, model = load_audio_model()
    output, sample_rate = generate_audio(processor, model, prompt, preset)
    return StreamingResponse(
        audio_array_to_buffer(output, sample_rate), media_type="audio/wav"
    )


@app.post("/users")
def create_user_controller(user: UserCreate):
    return {"name": user.username, "password": user.password}


@app.get("/", include_in_schema=False)
def docs_redirect_controller():
    return RedirectResponse(url="/docs", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/chat")
def chat_controller(prompt: str = "Inspire me"):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    statement = response.choices[0].message.content
    return {"statement": statement}


@app.get("/generate/text")
def serve_language_model_controller(prompt: str):
    pipe = load_text_model()
    output = generate_text(pipe, prompt)
    return output
