---
title: Ancient Court
emoji: 🏛️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
short_description: Chat and debate with a council of historical advisors
models:
  - Qwen/Qwen2.5-3B-Instruct
---

# Ancient Court

Ancient Court is a "council of advisors": chat with historical figures who
answer you in their own voices, react to each other's replies, and debate
among themselves. Replies are read aloud in the browser.

Current advisors:

| Advisor | Title | Character |
|---|---|---|
| Marcus Aurelius | Stoic Philosopher | Reflective, disciplined Stoic emperor |
| Niccolò Machiavelli | Florentine Diplomat | Shrewd, candid, dryly witty realist |
| Socrates | Athenian Philosopher | Humble, playful, answers with probing questions |
| Abraham Lincoln | 16th U.S. President | Plain-spoken, wry, makes points with frontier stories |

## How it works

All advisors share one instruction-tuned model, `Qwen/Qwen2.5-3B-Instruct`.
Each advisor's personality comes from its registry entry
(`src/persona/registry.py`), not from training:

```
Your message
  -> for each active advisor, in activation order:
       system prompt (persona + reply rules + "others present" line)
       + example exchanges (persona's few-shot examples)
       + conversation so far (other advisors' lines prefixed with their name)
  -> Qwen generates a reply (capped at ~120 tokens, trimmed to a full sentence)
  -> reply saved to the session, shown in the chat, and spoken aloud
```

- **System prompt:** who the advisor is, how they speak, a knowledge cutoff at
  their death, and a rule against writing code or other technical work.
- **Example exchanges:** a few short sample question/answer pairs per advisor,
  including rival exchanges with other advisors. Small models imitate these
  closely, so they shape length, tone, and how sharply advisors disagree.
- **Council awareness:** when several advisors are active, each one is told who
  else is present, that it is speaking as itself in the first person, and that
  the others are rivals, not friends.

A LoRA fine-tuning pipeline also exists (see
[Optional: fine-tuning an adapter](#optional-fine-tuning-an-adapter)), but it is
**off by default** because the current adapter reduced answer quality.

## Project structure

```
src/
  data_processing.py        # CLI: PDF -> JSON (only needed for fine-tuning)
  utils/pdf_utils.py        # PDF text extraction
  data_files/               # source PDFs + extracted JSON
  persona/
    registry.py             # PersonaConfig + all advisors (prompts, examples)
    infer.py                # model loading, prompt building, generation (CLI + functions)
    training_config.py      # base model id, LoRA + training hyperparameters
    dataset_builder.py      # JSON -> train/val JSONL (fine-tuning only)
    train.py                # LoRA fine-tuning CLI (fine-tuning only)
  training_data/            # generated train/val JSONL (gitignored)
  web/
    app.py                  # Flask app: entry point and chat routes
    conversation.py         # in-memory per-session transcript store
    templates/chat.html
    static/chat.js          # chat UI, debate button, browser speech
    static/style.css
checkpoints/                # trained LoRA adapters (gitignored)
tests/                      # pytest unit tests
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

**Model download:** the first reply downloads `Qwen/Qwen2.5-3B-Instruct`
(about 6 GB) into the Hugging Face cache at
`C:\Users\<you>\.cache\huggingface\hub\`. Set `HF_HOME` to use a different
location. In 16-bit it needs roughly 6–7 GB of GPU memory, so an 8 GB card is
enough. Symlink warnings from the Hugging Face cache on Windows are harmless.

> All commands below assume you're in the repo root (`ancient_court/`), since
> modules are imported as `src.persona.*` / `src.web.*`.

## Run the chat UI

```powershell
.venv\Scripts\python.exe -m flask --app src.web.app run
```

Then open `http://127.0.0.1:5000`.

- **Activate an advisor:** click their button. A spinner shows while the model
  loads (only slow the first time), then the advisor greets you.
- **Ask a question:** advisors reply one at a time, in the order you activated
  them, and each sees the replies given before it. Each reply appears and is
  spoken as soon as it's ready, while the next advisor's reply generates in the
  background (shown as "‹name› is thinking…").
- **Let them debate:** with two or more advisors active, this button runs a
  round in which each advisor responds to the others instead of to you. Click
  again for another round.
- **Voice:** replies are read aloud using the browser's built-in speech, with a
  different voice preference and pitch per advisor. Toggle it with the
  **Voice: On/Off** button. Edge has the most natural voices on Windows.

Add `--debug` to restart the server automatically when you save a `.py` file.
Without it, restart the server after changing any Python code or persona.

## Test a reply from the command line

```powershell
.venv\Scripts\python.exe -m src.persona.infer --persona marcus_aurelius --prompt "What is virtue?"
```

Omit `--checkpoint` to use the base model and persona prompt, which is the
same setup the chat UI uses.

## Customizing an advisor

Edit their entry in `src/persona/registry.py`:

- `system_prompt`: who they are, their voice, and their rules. Strings on
  separate lines are joined with no separator, so end each one with a space.
- `example_dialogue`: `(visitor line, advisor reply)` pairs shown before the
  real conversation. To show a reaction to another advisor, prefix the visitor
  line with that advisor's display name, e.g. `"Socrates: ..."`.

Shared rules for all advisors (reply length, debate instruction, "others
present" wording) live in `src/persona/infer.py`. Restart the server to apply
changes.

## Adding a new advisor

1. Add an entry to `PERSONAS` in `src/persona/registry.py` with a
   `persona_id`, `display_name`, `title`, `system_prompt`, and
   `example_dialogue`. Set `source_json=None` unless you plan to fine-tune.
2. Optionally add a voice preference and pitch for the new `persona_id` in
   `VOICE_SETTINGS` in `src/web/static/chat.js`. Without one, the browser's
   default English voice is used.
3. Restart the server. The advisor appears automatically as a button.

## Optional: fine-tuning an adapter

The original approach fine-tuned a LoRA adapter on a figure's writings. It's
kept for experiments but disabled in the chat UI, because training on book
passages paired with generic questions taught the model to recite rather than
answer. Only Marcus Aurelius has training data.

1. **Extract a PDF to JSON**

   ```powershell
   .venv\Scripts\python.exe src\data_processing.py src\data_files\Marcus-Aurelius-Meditations.pdf
   ```

2. **Build the training dataset.** Keeps only the twelve books (no
   introduction, appendix, or glossary), strips footnotes, section numbers,
   watermarks, and Greek text, adds short conversational examples, and writes
   `src/training_data/marcus_aurelius_train.jsonl` and `..._val.jsonl`.

   ```powershell
   .venv\Scripts\python.exe -m src.persona.dataset_builder marcus_aurelius
   ```

3. **Train**, saving the adapter to `checkpoints/marcus_aurelius/`:

   ```powershell
   .venv\Scripts\python.exe -m src.persona.train --persona marcus_aurelius
   ```

   Useful flags: `--max-steps N` for a quick smoke test, `--use-4bit` for
   4-bit QLoRA (unreliable on native Windows; prefer WSL2), and
   `--output-dir PATH` to save elsewhere.

4. **Use it in the chat UI:**

   ```powershell
   $env:PERSONA_USE_ADAPTER = "1"
   $env:MARCUS_AURELIUS_CHECKPOINT_DIR = "checkpoints/marcus_aurelius_smoke"  # optional override
   .venv\Scripts\python.exe -m flask --app src.web.app run
   ```

An adapter always loads the base model it was trained on (recorded in its
`adapter_config.json`). The existing `checkpoints/marcus_aurelius` adapter was
trained on `Qwen/Qwen2.5-1.5B-Instruct`. If you edit a persona's system
prompt, rebuild the dataset before retraining, since the JSONL files store
their own copy of it.

## Running tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Tests cover dataset building and conversation/prompt logic without requiring a
GPU or any model downloads.

## Known limitations

- **Chat history is in memory only.** It resets when the Flask server
  restarts; reload the page and reactivate advisors afterward.
- **The no-technical-work rule is unreliable.** Advisors sometimes still answer
  SQL or Excel questions despite the prompt rule.
- **Fixed speaking order.** Advisors always reply in activation order, so the
  first one never reacts to the others within a round.
- **Replies are sequential.** Advisors still generate one after another on the
  GPU, so a full round takes longer with more advisors, though you can read or
  listen to each reply while the next one generates.
- **Prompt rules slip sometimes.** A 3B model occasionally flatters other
  advisors, runs long, or drifts from its character.
- **Not deployable to GitHub Pages as-is.** The app needs a Python server and a
  GPU; GitHub Pages only serves static files.
