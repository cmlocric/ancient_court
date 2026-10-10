const sessionId = localStorage.getItem("session_id") || crypto.randomUUID();
localStorage.setItem("session_id", sessionId);

const transcriptEl = document.getElementById("transcript");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");
const personaChips = document.querySelectorAll(".persona-chip");
const voiceToggleEl = document.getElementById("voice-toggle");

const speechSupported = "speechSynthesis" in window;
let voiceEnabled = speechSupported && localStorage.getItem("voice_enabled") !== "false";
// Earlier names win; falls back to any English voice if none are installed.
const PREFERRED_VOICE_NAMES = ["Microsoft George", "Microsoft Ryan", "Google UK English Male", "Microsoft David", "Daniel"];

// Browsers load voices asynchronously; speaking before then falls back to a robotic default voice.
const voicesReady = new Promise((resolve) => {
  if (!speechSupported || window.speechSynthesis.getVoices().length) return resolve();
  window.speechSynthesis.addEventListener("voiceschanged", resolve, { once: true });
  setTimeout(resolve, 2000);
});

function pickVoice() {
  const voices = window.speechSynthesis.getVoices();
  for (const name of PREFERRED_VOICE_NAMES) {
    const match = voices.find((v) => v.name.includes(name));
    if (match) return match;
  }
  return voices.find((v) => v.lang.startsWith("en")) || null;
}

async function speak(text) {
  if (!voiceEnabled) return;
  await voicesReady;
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = pickVoice();
  if (voice) utterance.voice = voice;
  utterance.rate = 1.1;
  utterance.pitch = 0.85;
  // speechSynthesis queues utterances, so multiple advisors speak in turn rather than overlapping.
  window.speechSynthesis.speak(utterance);
}

function updateVoiceToggle() {
  voiceToggleEl.textContent = voiceEnabled ? "Voice: On" : "Voice: Off";
  voiceToggleEl.setAttribute("aria-pressed", String(voiceEnabled));
  voiceToggleEl.classList.toggle("voice-toggle-off", !voiceEnabled);
}

if (speechSupported) {
  voiceToggleEl.addEventListener("click", () => {
    voiceEnabled = !voiceEnabled;
    localStorage.setItem("voice_enabled", String(voiceEnabled));
    if (!voiceEnabled) window.speechSynthesis.cancel();
    updateVoiceToggle();
  });
  updateVoiceToggle();
} else {
  voiceToggleEl.hidden = true;
}

// Advisors whose model is loaded and who will respond to the next message.
const activePersonas = new Set();
// Only one activation (model load) runs at a time to avoid contending for the GPU/CPU.
let isActivating = false;

function setOtherChipsDisabled(currentChip, disabled) {
  personaChips.forEach((chip) => {
    if (chip !== currentChip && !activePersonas.has(chip.dataset.personaId)) {
      chip.disabled = disabled;
    }
  });
}

function renderTurn(turn) {
  const div = document.createElement("div");
  div.className = turn.speaker === "user" ? "message message-user" : "message message-persona";
  const speaker = document.createElement("span");
  speaker.className = "speaker";
  speaker.textContent = `${turn.speaker}:`;
  const text = document.createElement("span");
  text.textContent = turn.text;
  div.appendChild(speaker);
  div.appendChild(text);
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;

  if (turn.speaker !== "user") speak(turn.text);
}

async function handlePersonaChipClick(chip) {
  const personaId = chip.dataset.personaId;

  if (activePersonas.has(personaId)) {
    // No need to reload the model — just stop including this advisor in replies.
    activePersonas.delete(personaId);
    chip.classList.remove("persona-chip-active");
    return;
  }

  if (isActivating) return;
  isActivating = true;
  chip.classList.add("persona-chip-loading");
  chip.disabled = true;
  setOtherChipsDisabled(chip, true);

  try {
    const response = await fetch("/persona/activate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, persona_id: personaId }),
    });
    const data = await response.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    activePersonas.add(personaId);
    chip.classList.add("persona-chip-active");
    renderTurn(data.turn);
  } catch (err) {
    alert("Failed to activate advisor.");
  } finally {
    chip.classList.remove("persona-chip-loading");
    chip.disabled = false;
    setOtherChipsDisabled(chip, false);
    isActivating = false;
  }
}


personaChips.forEach((chip) => {
  chip.addEventListener("click", () => handlePersonaChipClick(chip));
});

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;

  inputEl.value = "";
  renderTurn({ speaker: "user", text: message });

  const response = await fetch("/chat/send", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      active_personas: Array.from(activePersonas),
    }),
  });

  const data = await response.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  // The user's own turn was already rendered immediately above; only render replies.
  data.turns.filter((turn) => turn.speaker !== "user").forEach(renderTurn);
});

