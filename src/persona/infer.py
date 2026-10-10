import argparse
import re
from pathlib import Path

from .registry import PERSONAS, get_persona
from .training_config import TrainingConfig

# Cache of loaded (model, tokenizer) per persona_id, keyed also by checkpoint dir
# so the web app and CLI both reuse already-loaded models in-process.
_MODEL_CACHE: dict[str, tuple] = {}
# Base models shared across personas when running without adapters, so each advisor doesn't load its own copy.
_BASE_MODEL_CACHE: dict[str, tuple] = {}


def _load_base_model(base_model_id: str):
    if base_model_id in _BASE_MODEL_CACHE:
        return _BASE_MODEL_CACHE[base_model_id]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    _BASE_MODEL_CACHE[base_model_id] = (model, tokenizer)
    return _BASE_MODEL_CACHE[base_model_id]


def load_persona_model(persona_id: str, checkpoint_dir: str | Path | None):
    """Load (and cache) a persona's model; with no checkpoint_dir, use the base model and persona prompt only."""
    persona = get_persona(persona_id)
    if checkpoint_dir is None:
        model, tokenizer = _load_base_model(TrainingConfig(persona_id=persona_id).base_model_id)
        return model, tokenizer, persona

    cache_key = f"{persona_id}:{checkpoint_dir}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_path = Path(checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Read the adapter config to find the original base model id it was trained from.
    import json

    adapter_config = json.loads((checkpoint_path / "adapter_config.json").read_text(encoding="utf-8"))
    base_model_id = adapter_config["base_model_name_or_path"]

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
    model.eval()

    _MODEL_CACHE[cache_key] = (model, tokenizer, persona)
    return _MODEL_CACHE[cache_key]


# Keeps replies short and on-topic; also used verbatim in training so train/inference prompts match.
_CONCISENESS_HINT = (
    "Always respond directly to what was just said to you, "
    "concisely, in no more than two or three sentences."
)

# Used to prompt a persona's opening line when it's first activated in a conversation.
_GREETING_INSTRUCTION = "Introduce yourself and offer a brief greeting to those gathered."


def build_messages_for_persona(persona, transcript: list[dict]) -> list[dict]:
    """Build chat-template messages from the persona's point of view."""
    messages = [{"role": "system", "content": f"{persona.system_prompt} {_CONCISENESS_HINT}"}]
    for user_text, persona_text in persona.example_dialogue:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": persona_text})
    for turn in transcript:
        speaker = turn["speaker"]
        if speaker == persona.persona_id:
            messages.append({"role": "assistant", "content": turn["text"]})
        elif speaker == "user":
            messages.append({"role": "user", "content": turn["text"]})
        else:
            # Other advisors are presented as named speakers so this persona can react to them.
            other = PERSONAS.get(speaker)
            name = other.display_name if other else speaker
            messages.append({"role": "user", "content": f"{name}: {turn['text']}"})
    return messages


def _run_generation(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return _trim_to_last_sentence(tokenizer.decode(generated, skip_special_tokens=True).strip())


_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*")


def _trim_to_last_sentence(text: str) -> str:
    # Drops a trailing fragment left when generation hits max_new_tokens mid-sentence.
    ends = list(_SENTENCE_END_RE.finditer(text))
    return text[: ends[-1].end()] if ends else text


def generate(
    persona_id: str,
    transcript: list[dict],
    checkpoint_dir: str | Path | None,
    max_new_tokens: int = 120,
) -> str:
    """Generate a persona-voiced reply continuing the given conversation transcript."""
    model, tokenizer, persona = load_persona_model(persona_id, checkpoint_dir)
    messages = build_messages_for_persona(persona, transcript)
    return _run_generation(model, tokenizer, messages, max_new_tokens)


def generate_greeting(
    persona_id: str,
    transcript: list[dict],
    checkpoint_dir: str | Path | None,
    max_new_tokens: int = 80,
) -> str:
    """Generate a persona's opening greeting, without persisting the prompt instruction as a real turn."""
    model, tokenizer, persona = load_persona_model(persona_id, checkpoint_dir)
    greeting_transcript = transcript + [{"speaker": "user", "text": _GREETING_INSTRUCTION}]
    messages = build_messages_for_persona(persona, greeting_transcript)
    return _run_generation(model, tokenizer, messages, max_new_tokens)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a persona-voiced response.")
    parser.add_argument("--persona", required=True, help="Persona id from the registry.")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained LoRA checkpoint; omit to use the base model with the persona prompt.")
    parser.add_argument("--prompt", required=True, help="User prompt/question to respond to.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    reply = generate(
        persona_id=args.persona,
        transcript=[{"speaker": "user", "text": args.prompt}],
        checkpoint_dir=args.checkpoint,
    )
    print(reply)

