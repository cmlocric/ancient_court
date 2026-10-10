import argparse
from pathlib import Path

from .registry import get_persona

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
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
    model.eval()

    _MODEL_CACHE[cache_key] = (model, tokenizer, persona)
    return _MODEL_CACHE[cache_key]


# Keeps replies short without a hard token cutoff; models otherwise ramble on indefinitely.
_CONCISENESS_HINT = "Respond concisely, in no more than two or three sentences."


def build_prompt_for_persona(persona, transcript: list[dict]) -> str:
    """Build a generation prompt from a persona's system prompt + prior conversation turns."""
    lines = [f"<|system|>\n{persona.system_prompt} {_CONCISENESS_HINT}"]
    for turn in transcript:
        lines.append(f"<|{turn['speaker']}|>\n{turn['text']}")
    lines.append(f"<|{persona.persona_id}|>\n")
    return "\n".join(lines)


def generate(
    persona_id: str,
    transcript: list[dict],
    checkpoint_dir: str | Path,
    max_new_tokens: int = 300,
) -> str:
    """Generate a persona-voiced reply continuing the given conversation transcript."""
    import torch

    model, tokenizer, persona = load_persona_model(persona_id, checkpoint_dir)
    prompt = build_prompt_for_persona(persona, transcript)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            # Stop as soon as the model starts hallucinating the next turn,
            # instead of running on until max_new_tokens truncates it mid-sentence.
            stop_strings=["<|"],
            tokenizer=tokenizer,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Safety net in case the stop string lands mid-token and isn't caught above.
    marker_index = text.find("<|")
    if marker_index != -1:
        text = text[:marker_index].strip()
    return text


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
