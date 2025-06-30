from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse
from openai import OpenAI
from pydantic import BaseModel, field_validator


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


app.post("/users")


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
