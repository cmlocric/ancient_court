# Ancient Court

Ancient Court turns historical texts into chattable personas. It fine-tunes a
lightweight pretrained LLM (LoRA/PEFT) on a historical figure's writings so it
responds in that figure's voice, then serves the result through a simple Flask
chat UI. The long-term goal is a "council of advisors" — multiple personas
active in one conversation, each aware of what the others have said.

Currently seeded with **Marcus Aurelius** (Roman Emperor, Stoic philosopher),
built from the public-domain *Meditations* (Casaubon translation).

## How it works

```
PDF (data_files/)
  -> JSON page extraction        (src/utils/pdf_utils.py, src/data_processing.py)
  -> instruction/response JSONL  (src/persona/dataset_builder.py)
  -> LoRA fine-tuning             (src/persona/train.py)
  -> adapter checkpoint          (checkpoints/{persona_id}/)
  -> generation                  (src/persona/infer.py)
  -> chat UI                     (src/web/app.py, Flask)
```

Personas are registry-driven (`src/persona/registry.py`), so adding a new
persona later just means adding a source JSON + one registry entry — no
changes needed to the dataset builder, trainer, or web app.

## Project structure

```
src/
  app.py                 # legacy entry point
  data_processing.py      # CLI: PDF -> JSON
  utils/pdf_utils.py       # PDF text extraction
  data_files/              # source PDFs + extracted JSON
  persona/
    registry.py            # PersonaConfig + persona registry
    dataset_builder.py      # JSON -> instruction/response JSONL
    training_config.py      # TrainingConfig dataclass (model, LoRA, hyperparams)
    train.py                # LoRA fine-tuning CLI (peft + trl.SFTTrainer)
    infer.py                # load adapter + generate (CLI + reusable functions)
  training_data/            # generated train/val JSONL (gitignored)
  web/
    app.py                  # Flask app (chat routes)
    conversation.py          # in-memory per-session transcript store
    templates/chat.html
    static/chat.js, style.css
checkpoints/               # trained LoRA adapters (gitignored)
tests/                     # pytest unit tests
requirements.txt
```

## Setup

```powershell
cd C:\Users\15864\GItHub\ancient_court
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**GPU note (Windows):** plain `pip install torch` installs a CPU-only build.
For CUDA/GPU support, install the matching CUDA build afterward, e.g.:

```powershell
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch --upgrade
```

Verify it worked:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

> All commands below assume you're in the repo root (`ancient_court/`), since
> modules are imported as `src.persona.*` / `src.web.*`.

## 1. Extract a PDF to JSON

```powershell
.venv\Scripts\python.exe src\data_processing.py src\data_files\Marcus-Aurelius-Meditations.pdf
```

Produces a `.json` file alongside the PDF with `{"source": ..., "pages": [{"page": N, "text": "..."}]}`.

## 2. Build the training dataset

```powershell
.venv\Scripts\python.exe -m src.persona.dataset_builder marcus_aurelius
```

Chunks the persona's source JSON into passages, pairs each with a reflective
instruction prompt, and writes `src/training_data/marcus_aurelius_train.jsonl`
and `..._val.jsonl` (90/10 split).

## 3. Fine-tune the persona

```powershell
.venv\Scripts\python.exe -m src.persona.train --persona marcus_aurelius
```

Fine-tunes the base model (`microsoft/phi-2` by default) with LoRA and saves
the adapter to `checkpoints/marcus_aurelius/`.

Useful flags:
- `--max-steps N` — cap training steps (e.g. for a quick smoke test)
- `--use-4bit` — enable 4-bit QLoRA (bitsandbytes; unreliable on native
  Windows, prefer WSL2 if you need it)
- `--output-dir PATH` — save the checkpoint somewhere other than
  `checkpoints/{persona_id}`

## 4. Generate a response from the command line

```powershell
.venv\Scripts\python.exe -m src.persona.infer --persona marcus_aurelius --checkpoint checkpoints\marcus_aurelius --prompt "What is virtue?"
```

## 5. Run the chat UI

```powershell
.venv\Scripts\python.exe -m flask --app src.web.app run
```

Then open `http://127.0.0.1:5000`. Select one or more personas and start
chatting — each active persona sees the full conversation so far, including
other personas' replies in the same round.

By default each persona loads its checkpoint from `checkpoints/{persona_id}`.
To point a persona at a different checkpoint (e.g. a smoke-test run), set an
environment variable before starting the server:

```powershell
$env:MARCUS_AURELIUS_CHECKPOINT_DIR = "checkpoints/marcus_aurelius_smoke"
.venv\Scripts\python.exe -m flask --app src.web.app run
```

## Running tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Tests cover dataset building and conversation/prompt logic without requiring a
GPU or any model downloads.

## Adding a new persona

1. Add a source JSON (same `{"source", "pages": [{"page", "text"}]}` shape —
   use `data_processing.py` to produce one from a PDF).
2. Add an entry to `PERSONAS` in `src/persona/registry.py` with a
   `persona_id`, `display_name`, `title`, `source_json` path, and
   `system_prompt`.
3. Run the dataset builder and trainer for that `persona_id`.
4. It will automatically appear as a selectable persona in the chat UI.

## Known limitations (POC stage)

- Chat history is in-memory only — it resets when the Flask server restarts.
- Dataset instructions are generic/templated, not topic-aware.
- Only one persona is trained so far; multi-persona "council" behavior is
  architected but not yet exercised with more than one trained checkpoint.
