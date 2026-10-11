import html
import json
from pathlib import Path

import spaces  # Must be imported before CUDA is touched on ZeroGPU Spaces.
import gradio as gr

from src.persona.infer import generate, generate_debate_reply, generate_greeting, load_persona_model
from src.persona.registry import PERSONAS

# ZeroGPU requires models to be placed on CUDA at startup; all advisors share this base model.
load_persona_model(next(iter(PERSONAS)), None)

_GENERATORS = {"reply": generate, "debate": generate_debate_reply, "greeting": generate_greeting}


@spaces.GPU(duration=60)
def _generate_turn(kind: str, persona_id: str, transcript: list[dict], present: list[str]) -> str:
    return _GENERATORS[kind](persona_id, transcript, None, present_persona_ids=present)


def _name(persona_id: str) -> str:
    return PERSONAS[persona_id].display_name


def _speech_payload(persona_id: str, text: str, counter: int) -> str:
    # The counter makes every payload unique so the browser speaks repeated text too.
    return json.dumps({"speaker": persona_id, "text": text, "n": counter})


def _render_transcript(transcript: list[dict], thinking: str | None = None) -> str:
    """Build the conversation HTML using the same markup and classes as the Flask UI."""
    parts = []
    for turn in transcript:
        is_user = turn["speaker"] == "user"
        css_class = "message message-user" if is_user else "message message-persona"
        name = "You" if is_user else _name(turn["speaker"])
        parts.append(
            f'<div class="{css_class}"><span class="speaker">{html.escape(name)}:</span>'
            f"<span>{html.escape(turn['text'])}</span></div>"
        )
    if thinking:
        parts.append(
            f'<div class="message message-persona message-thinking">{html.escape(_name(thinking))} is thinking…</div>'
        )
    # Newest first: #transcript is a reversed column, which keeps the latest message scrolled into view.
    parts.reverse()
    return f'<div id="transcript">{"".join(parts)}</div>'


def _run_turns(kind, persona_ids, present, transcript, speech_count):
    """Generate one advisor at a time, yielding after each so replies appear (and are spoken) as they finish."""
    for persona_id in persona_ids:
        yield transcript, _render_transcript(transcript, thinking=persona_id), gr.skip(), speech_count

        reply = _generate_turn(kind, persona_id, transcript, present)
        transcript = transcript + [{"speaker": persona_id, "text": reply}]
        speech_count += 1
        yield transcript, _render_transcript(transcript), _speech_payload(persona_id, reply, speech_count), speech_count


def on_advisors_change(selected, active, transcript, speech_count):
    selected = selected or []
    # Keep activation order: existing advisors first, newcomers appended as they join.
    active = [pid for pid in active if pid in selected] + [pid for pid in selected if pid not in active]
    yield active, transcript, gr.skip(), gr.skip(), speech_count

    already_spoke = {turn["speaker"] for turn in transcript}
    for persona_id in [pid for pid in active if pid not in already_spoke]:
        present = active[: active.index(persona_id) + 1]
        for transcript, rendered, speech, speech_count in _run_turns(
            "greeting", [persona_id], present, transcript, speech_count
        ):
            yield active, transcript, rendered, speech, speech_count


def on_send(message, active, transcript, speech_count):
    message = (message or "").strip()
    if not message:
        return
    if not active:
        gr.Warning("Select at least one advisor first.")
        return

    transcript = transcript + [{"speaker": "user", "text": message}]
    yield "", transcript, _render_transcript(transcript), gr.skip(), speech_count

    for transcript, rendered, speech, speech_count in _run_turns("reply", active, active, transcript, speech_count):
        yield "", transcript, rendered, speech, speech_count


def on_debate(active, transcript, speech_count):
    if len(active) < 2:
        gr.Warning("A debate needs at least two advisors.")
        return
    yield from _run_turns("debate", active, active, transcript, speech_count)


def on_voice_toggle(voice_on):
    voice_on = not voice_on
    label = "Voice: On" if voice_on else "Voice: Off"
    return voice_on, gr.Button(value=label, elem_classes=[] if voice_on else ["voice-toggle-off"])


# Browser speech: same per-advisor voice preferences as the Flask UI's chat.js.
_SPEAK_JS = """
(payload, voiceOn) => {
  if (!voiceOn || !payload || !("speechSynthesis" in window)) return;
  const { speaker, text } = JSON.parse(payload);
  const settings = {
    marcus_aurelius: { names: ["Microsoft George", "Microsoft Ryan", "Google UK English Male", "Microsoft David", "Daniel"], pitch: 0.85 },
    machiavelli: { names: ["Microsoft Guy", "Microsoft Mark", "Google US English", "Microsoft David", "Alex"], pitch: 1.05 },
    socrates: { names: ["Microsoft Ryan", "Microsoft George", "Google UK English Male", "Microsoft David", "Daniel"], pitch: 1.15 },
    abraham_lincoln: { names: ["Microsoft Christopher", "Microsoft Eric", "Microsoft David", "Google US English", "Alex"], pitch: 0.75 },
  }[speaker] || { names: [], pitch: 1.0 };
  const speakNow = () => {
    const voices = window.speechSynthesis.getVoices();
    let voice = null;
    for (const name of settings.names) {
      voice = voices.find((v) => v.name.includes(name));
      if (voice) break;
    }
    voice = voice || voices.find((v) => v.lang.startsWith("en")) || null;
    const utterance = new SpeechSynthesisUtterance(text);
    if (voice) utterance.voice = voice;
    utterance.rate = 1.1;
    utterance.pitch = settings.pitch;
    window.speechSynthesis.speak(utterance);
  };
  if (window.speechSynthesis.getVoices().length) speakNow();
  else window.speechSynthesis.addEventListener("voiceschanged", speakNow, { once: true });
}
"""

_STOP_SPEECH_JS = "(voiceOn) => { if (!voiceOn && 'speechSynthesis' in window) window.speechSynthesis.cancel(); }"

# Shared with the Flask UI so both look the same.
_STYLE_CSS = Path(__file__).parent / "src" / "web" / "static" / "style.css"

# Neutralizes Gradio's own wrappers so the shared stylesheet shows through.
_GRADIO_OVERRIDES_CSS = """
.gradio-container { background: transparent !important; max-width: 1040px !important; width: 100% !important; margin: 0 auto !important; }
/* Restores the shared stylesheet's navy band behind the header, which Gradio's page background covers. */
body, gradio-app {
  background: linear-gradient(180deg, var(--navy) 0%, var(--navy-light) 220px, #f2efe8 220px) !important;
}
.gradio-container main, .gradio-container .contain { max-width: 1040px !important; width: 100% !important; }
.gradio-container .chat-card { max-width: 980px !important; width: 100% !important; margin: 0 auto 2.5rem !important; }
.gradio-container .block, .gradio-container .form { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
footer { display: none !important; }

/* Gradio's heading styles otherwise shrink the title to body-text size. */
header h1 { font-size: 2.6rem !important; line-height: 1.2 !important; font-weight: 700 !important; }
header .subtitle { font-size: 1rem !important; }

/* Newest message at the bottom and kept in view: turns are rendered newest-first in a reversed column. */
#transcript { display: flex; flex-direction: column-reverse; justify-content: flex-end; height: 480px; }

#personas { gap: 0.6rem !important; }
#advisors .wrap { gap: 0.6rem; }
#advisors label {
  background: #fff !important; border: 1px solid var(--border) !important; border-radius: 999px !important;
  padding: 0.35rem 0.75rem !important; font-size: 0.9rem; color: var(--ink) !important; cursor: pointer;
}
#advisors label:has(input:checked), #advisors label.selected { border-color: var(--gold) !important; background: #fdf6e6 !important; }
#advisors input[type="checkbox"] { display: none; }

#voice-toggle { flex: 0 0 auto !important; min-width: 0 !important; }
#voice-toggle.voice-toggle-off { background: #fff !important; color: var(--muted) !important; border-color: var(--border) !important; }

#message-input textarea, #message-input input {
  padding: 0.65rem 0.9rem !important; border: 1px solid var(--border) !important; border-radius: 8px !important;
  font-size: 0.95rem !important; background: #fff !important; color: var(--ink) !important; box-shadow: none !important;
}
#message-input textarea:focus, #message-input input:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 3px rgba(185, 146, 47, 0.2) !important; }
#chat-form { align-items: stretch; }
#chat-form button { flex: 0 0 auto !important; min-width: 0 !important; }
"""

# The shared stylesheet is light-only, so keep Gradio from switching to dark mode.
_FORCE_LIGHT_JS = "() => { document.querySelectorAll('.dark').forEach((el) => el.classList.remove('dark')); }"

_HEADER_HTML = (
    "<header><h1>Ancient Court</h1>"
    "<p class=\"subtitle\">Converse with history's great advisors</p></header>"
)


with gr.Blocks(title="Ancient Court") as demo:
    active_state = gr.State([])
    transcript_state = gr.State([])
    speech_count_state = gr.State(0)

    gr.HTML(_HEADER_HTML, apply_default_css=False)

    with gr.Column(elem_classes=["chat-card"]):
        with gr.Row(elem_id="personas"):
            gr.HTML('<span class="section-label">Advisors</span>', apply_default_css=False, scale=0, min_width=0)
            advisors = gr.CheckboxGroup(
                choices=[(f"{p.display_name} — {p.title}", p.persona_id) for p in PERSONAS.values()],
                show_label=False,
                container=False,
                elem_id="advisors",
                scale=1,
            )
            voice_button = gr.Button("Voice: On", elem_id="voice-toggle", size="sm", scale=0)

        transcript_html = gr.HTML(_render_transcript([]), apply_default_css=False)

        with gr.Row(elem_id="chat-form"):
            message = gr.Textbox(
                placeholder="Speak to the court...", show_label=False, container=False, elem_id="message-input", scale=1
            )
            send = gr.Button("Send", elem_id="send-button", scale=0)
            debate = gr.Button("Let them debate", elem_id="debate-button", scale=0)

    # Hidden channels: voice on/off for the browser, and each finished reply to be spoken.
    voice_on = gr.Checkbox(value=True, visible=False)
    speech = gr.Textbox(visible=False)

    advisors.change(
        on_advisors_change,
        inputs=[advisors, active_state, transcript_state, speech_count_state],
        outputs=[active_state, transcript_state, transcript_html, speech, speech_count_state],
        show_progress="hidden",
    )
    send_inputs = [message, active_state, transcript_state, speech_count_state]
    send_outputs = [message, transcript_state, transcript_html, speech, speech_count_state]
    send.click(on_send, inputs=send_inputs, outputs=send_outputs, show_progress="hidden")
    message.submit(on_send, inputs=send_inputs, outputs=send_outputs, show_progress="hidden")
    debate.click(
        on_debate,
        inputs=[active_state, transcript_state, speech_count_state],
        outputs=[transcript_state, transcript_html, speech, speech_count_state],
        show_progress="hidden",
    )
    voice_button.click(on_voice_toggle, inputs=[voice_on], outputs=[voice_on, voice_button], show_progress="hidden")
    speech.change(None, inputs=[speech, voice_on], js=_SPEAK_JS)
    voice_on.change(None, inputs=[voice_on], js=_STOP_SPEECH_JS)


if __name__ == "__main__":
    demo.launch(css_paths=[_STYLE_CSS], css=_GRADIO_OVERRIDES_CSS, js=_FORCE_LIGHT_JS)
