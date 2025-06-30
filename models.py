import numpy as np
import torch
from transformers import AutoModel, AutoProcessor, BarkModel, BarkProcessor
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline

from schemas import VoicePresets

prompt = "How to setup a FastAPI project?"
system_prompt = """
Your name is FastAPI bot and you are a useful chatbot responsible for teaching FastAPI to your users.
Always respond in Markdown.
"""

device = torch.device("cpu")


def load_text_model():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device=device,
    )
    return pipe


def generate_text(pipe: Pipeline, prompt: str, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    if pipe.tokenizer is None:
        raise ValueError("Pipeline tokenizer is None")

    chat_prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if not isinstance(chat_prompt, str):
        raise ValueError("Chat template did not return a string")

    predictions = pipe(
        chat_prompt,
        temperature=temperature,
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

    output = predictions[0]["generated_text"].split("</s>\n<|assistant|>\n")[-1]
    return output


def load_audio_model() -> tuple[BarkProcessor, BarkModel]:
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small").to(device)
    return processor, model


def generate_audio(
    processor: BarkProcessor, model: BarkModel, prompt: str, preset: VoicePresets
) -> tuple[np.array, int]:
    inputs = processor(text=[prompt], return_tensors="pt", voice_preset=preset)
    output = model.generate(**inputs, do_sample=True).cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    return output, sample_rate
