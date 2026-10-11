import json

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


def _run_turns(kind, persona_ids, present, transcript, chat, speech_count):
    """Generate one advisor at a time, yielding after each so replies appear (and are spoken) as they finish."""
    for persona_id in persona_ids:
        chat = chat + [{"role": "assistant", "content": f"_{_name(persona_id)} is thinking…_"}]
        yield transcript, chat, gr.skip(), speech_count

        reply = _generate_turn(kind, persona_id, transcript, present)
        transcript = transcript + [{"speaker": persona_id, "text": reply}]
        chat = chat[:-1] + [{"role": "assistant", "content": f"**{_name(persona_id)}:** {reply}"}]
        speech_count += 1
        yield transcript, chat, _speech_payload(persona_id, reply, speech_count), speech_count


def on_advisors_change(selected, active, transcript, chat, speech_count):
    selected = selected or []
    # Keep activation order: existing advisors first, newcomers appended as they join.
    active = [pid for pid in active if pid in selected] + [pid for pid in selected if pid not in active]
    yield active, transcript, chat, gr.skip(), speech_count

    already_spoke = {turn["speaker"] for turn in transcript}
    for persona_id in [pid for pid in active if pid not in already_spoke]:
        present = active[: active.index(persona_id) + 1]
        for transcript, chat, speech, speech_count in _run_turns(
            "greeting", [persona_id], present, transcript, chat, speech_count
        ):
            yield active, transcript, chat, speech, speech_count


def on_send(message, active, transcript, chat, speech_count):
    message = (message or "").strip()
    if not message:
        return
    if not active:
        gr.Warning("Select at least one advisor first.")
        return

    transcript = transcript + [{"speaker": "user", "text": message}]
    chat = chat + [{"role": "user", "content": message}]
    yield "", transcript, chat, gr.skip(), speech_count

    for transcript, chat, speech, speech_count in _run_turns(
        "reply", active, active, transcript, chat, speech_count
    ):
        yield "", transcript, chat, speech, speech_count


def on_debate(active, transcript, chat, speech_count):
    if len(active) < 2:
        gr.Warning("A debate needs at least two advisors.")
        return
    yield from _run_turns("debate", active, active, transcript, chat, speech_count)


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


with gr.Blocks(title="Ancient Court") as demo:
    gr.Markdown("# Ancient Court\n*Converse with history's great advisors.*")

    active_state = gr.State([])
    transcript_state = gr.State([])
    speech_count_state = gr.State(0)

    with gr.Row():
        advisors = gr.CheckboxGroup(
            choices=[(f"{p.display_name} — {p.title}", p.persona_id) for p in PERSONAS.values()],
            label="Advisors (select to invite them; each greets you on arrival)",
            scale=4,
        )
        voice_on = gr.Checkbox(value=True, label="Read replies aloud", scale=1)

    chatbot = gr.Chatbot(height=480, label="The court")
    message = gr.Textbox(placeholder="Speak to the court...", show_label=False)
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        debate = gr.Button("Let them debate")

    # Hidden channel carrying each finished reply to the browser for speech.
    speech = gr.Textbox(visible=False)

    advisors.change(
        on_advisors_change,
        inputs=[advisors, active_state, transcript_state, chatbot, speech_count_state],
        outputs=[active_state, transcript_state, chatbot, speech, speech_count_state],
    )
    send_inputs = [message, active_state, transcript_state, chatbot, speech_count_state]
    send_outputs = [message, transcript_state, chatbot, speech, speech_count_state]
    send.click(on_send, inputs=send_inputs, outputs=send_outputs)
    message.submit(on_send, inputs=send_inputs, outputs=send_outputs)
    debate.click(
        on_debate,
        inputs=[active_state, transcript_state, chatbot, speech_count_state],
        outputs=[transcript_state, chatbot, speech, speech_count_state],
    )
    speech.change(None, inputs=[speech, voice_on], js=_SPEAK_JS)
    voice_on.change(None, inputs=[voice_on], js=_STOP_SPEECH_JS)


if __name__ == "__main__":
    demo.launch()
