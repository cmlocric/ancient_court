import argparse
from pathlib import Path

from .registry import PERSONAS, get_persona

# Cache of loaded (model, tokenizer) per persona_id, keyed also by checkpoint dir
# so the web app and CLI both reuse already-loaded models in-process.
_MODEL_CACHE: dict[str, tuple] = {}


def load_persona_model(persona_id: str, checkpoint_dir: str | Path):
    """Load (and cache) a persona's base model + LoRA adapter for generation."""
    cache_key = f"{persona_id}:{checkpoint_dir}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    persona = get_persona(persona_id)
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
    "concisely, in no more than three or four sentences."
)

# Used to prompt a persona's opening line when it's first activated in a conversation.
_GREETING_INSTRUCTION = "Introduce yourself and offer a brief greeting to those gathered."


def build_messages_for_persona(persona, transcript: list[dict]) -> list[dict]:
    """Build chat-template messages from the persona's point of view."""
    messages = [{"role": "system", "content": f"{persona.system_prompt} {_CONCISENESS_HINT}"}]
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
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate(
    persona_id: str,
    transcript: list[dict],
    checkpoint_dir: str | Path,
    max_new_tokens: int = 300,
) -> str:
    """Generate a persona-voiced reply continuing the given conversation transcript."""
    model, tokenizer, persona = load_persona_model(persona_id, checkpoint_dir)
    messages = build_messages_for_persona(persona, transcript)
    return _run_generation(model, tokenizer, messages, max_new_tokens)


def generate_greeting(
    persona_id: str,
    transcript: list[dict],
    checkpoint_dir: str | Path,
    max_new_tokens: int = 150,
) -> str:
    """Generate a persona's opening greeting, without persisting the prompt instruction as a real turn."""
    model, tokenizer, persona = load_persona_model(persona_id, checkpoint_dir)
    greeting_transcript = transcript + [{"speaker": "user", "text": _GREETING_INSTRUCTION}]
    messages = build_messages_for_persona(persona, greeting_transcript)
    return _run_generation(model, tokenizer, messages, max_new_tokens)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a persona-voiced response.")
    parser.add_argument("--persona", required=True, help="Persona id from the registry.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained LoRA checkpoint directory.")
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

