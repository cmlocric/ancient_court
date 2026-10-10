const sessionId = localStorage.getItem("session_id") || crypto.randomUUID();
localStorage.setItem("session_id", sessionId);

const transcriptEl = document.getElementById("transcript");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");
const personaChips = document.querySelectorAll(".persona-chip");
const voiceToggleEl = document.getElementById("voice-toggle");
const sendButtonEl = document.getElementById("send-button");
const debateButtonEl = document.getElementById("debate-button");

const displayNames = Object.fromEntries(
  Array.from(personaChips).map((chip) => [chip.dataset.personaId, chip.dataset.displayName])
);

const speechSupported = "speechSynthesis" in window;
let voiceEnabled = speechSupported && localStorage.getItem("voice_enabled") !== "false";
// Per-advisor voice preferences; earlier names win. Different pitch keeps advisors distinct even if they share a voice.
const VOICE_SETTINGS = {
  marcus_aurelius: {
    names: ["Microsoft George", "Microsoft Ryan", "Google UK English Male", "Microsoft David", "Daniel"],
    pitch: 0.85,
  },
  machiavelli: {
    names: ["Microsoft Guy", "Microsoft Mark", "Google US English", "Microsoft David", "Alex"],
    pitch: 1.05,
  },
};
const DEFAULT_VOICE_SETTINGS = { names: [], pitch: 1.0 };

// Browsers load voices asynchronously; speaking before then falls back to a robotic default voice.
const voicesReady = new Promise((resolve) => {
  if (!speechSupported || window.speechSynthesis.getVoices().length) return resolve();
  window.speechSynthesis.addEventListener("voiceschanged", resolve, { once: true });
  setTimeout(resolve, 2000);
});

function pickVoice(preferredNames) {
  const voices = window.speechSynthesis.getVoices();
  for (const name of preferredNames) {
    const match = voices.find((v) => v.name.includes(name));
    if (match) return match;
  }
  return voices.find((v) => v.lang.startsWith("en")) || null;
}

async function speak(text, speakerId) {
  if (!voiceEnabled) return;
  await voicesReady;
  const settings = VOICE_SETTINGS[speakerId] || DEFAULT_VOICE_SETTINGS;
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = pickVoice(settings.names);
  if (voice) utterance.voice = voice;
  utterance.rate = 1.1;
  utterance.pitch = settings.pitch;
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
// True while waiting on a send or debate request, so rounds can't overlap.
let isBusy = false;

function updateActionButtons() {
  sendButtonEl.disabled = isBusy;
  debateButtonEl.disabled = isBusy || activePersonas.size < 2;
}

async function postTurns(url, body) {
  isBusy = true;
  updateActionButtons();
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    data.turns.filter((turn) => turn.speaker !== "user").forEach(renderTurn);
  } catch (err) {
    alert("Request failed.");
  } finally {
    isBusy = false;
    updateActionButtons();
  }
}

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
  speaker.textContent = turn.speaker === "user" ? "You:" : `${displayNames[turn.speaker] || turn.speaker}:`;
  const text = document.createElement("span");
  text.textContent = turn.text;
  div.appendChild(speaker);
  div.appendChild(text);
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;

  if (turn.speaker !== "user") speak(turn.text, turn.speaker);
}

async function handlePersonaChipClick(chip) {
  const personaId = chip.dataset.personaId;

  if (activePersonas.has(personaId)) {
    // No need to reload the model — just stop including this advisor in replies.
    activePersonas.delete(personaId);
    chip.classList.remove("persona-chip-active");
    updateActionButtons();
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
      body: JSON.stringify({
        session_id: sessionId,
        persona_id: personaId,
        active_personas: Array.from(activePersonas),
      }),
    });
    const data = await response.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    activePersonas.add(personaId);
    chip.classList.add("persona-chip-active");
    updateActionButtons();
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
  if (!message || isBusy) return;

  inputEl.value = "";
  renderTurn({ speaker: "user", text: message });

  await postTurns("/chat/send", {
    session_id: sessionId,
    message,
    active_personas: Array.from(activePersonas),
  });
});

debateButtonEl.addEventListener("click", async () => {
  if (isBusy || activePersonas.size < 2) return;
  await postTurns("/chat/debate", {
    session_id: sessionId,
    active_personas: Array.from(activePersonas),
  });
});

