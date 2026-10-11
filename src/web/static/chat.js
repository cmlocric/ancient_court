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
    gender: "male",
    names: ["Microsoft George", "Microsoft Ryan", "Google UK English Male", "Microsoft David", "Daniel"],
    pitch: 0.85,
  },
  machiavelli: {
    gender: "male",
    names: ["Microsoft Guy", "Microsoft Mark", "Microsoft David", "Alex", "Aaron"],
    pitch: 1.05,
  },
  socrates: {
    gender: "male",
    names: ["Microsoft Ryan", "Microsoft George", "Google UK English Male", "Microsoft David", "Arthur"],
    pitch: 1.15,
  },
  abraham_lincoln: {
    gender: "male",
    names: ["Microsoft Christopher", "Microsoft Eric", "Microsoft David", "Alex", "Fred"],
    pitch: 0.75,
  },
};
const DEFAULT_VOICE_SETTINGS = { gender: "male", names: [], pitch: 1.0 };

// Browsers don't expose a voice's gender, so it's inferred from known voice names on Windows, Apple and Android.
const VOICE_GENDER_HINTS = {
  male: [
    " male", "david", "mark", "george", "ryan", "guy", "christopher", "eric", "daniel", "alex", "aaron",
    "arthur", "fred", "gordon", "rishi", "ralph", "albert", "reed", "thomas", "oliver", "james",
    "-x-iom", "-x-iol", "-x-tpd", "-x-rjs", "-x-gbd",
  ],
  female: [
    "female", "zira", "samantha", "karen", "moira", "tessa", "martha", "catherine", "nicky", "victoria",
    "susan", "hazel", "aria", "jenny", "libby", "sonia", "emma", "google us english",
    "-x-iob", "-x-iog", "-x-sfg", "-x-tpc", "-x-tpf",
  ],
};
// Extra pitch drop when no voice of the right gender is installed, so the fallback at least sounds deeper.
const MISMATCHED_VOICE_PITCH_FACTOR = 0.75;

// Browsers load voices asynchronously; speaking before then falls back to a robotic default voice.
const voicesReady = new Promise((resolve) => {
  if (!speechSupported || window.speechSynthesis.getVoices().length) return resolve();
  window.speechSynthesis.addEventListener("voiceschanged", resolve, { once: true });
  setTimeout(resolve, 2000);
});

function voiceMatchesGender(voice, gender) {
  const label = ` ${voice.name} ${voice.voiceURI}`.toLowerCase();
  const other = gender === "male" ? "female" : "male";
  if (VOICE_GENDER_HINTS[other].some((hint) => label.includes(hint))) return false;
  return VOICE_GENDER_HINTS[gender].some((hint) => label.includes(hint));
}

function voiceIsOtherGender(voice, gender) {
  const label = ` ${voice.name} ${voice.voiceURI}`.toLowerCase();
  const other = gender === "male" ? "female" : "male";
  return VOICE_GENDER_HINTS[other].some((hint) => label.includes(hint));
}

// Returns { voice, genderMatched }: preferred names first, then any voice of the right gender, then any non-mismatched voice.
function pickVoice(settings) {
  const voices = window.speechSynthesis.getVoices().filter((v) => v.lang.toLowerCase().startsWith("en"));
  for (const name of settings.names) {
    const match = voices.find((v) => v.name.includes(name) && !voiceIsOtherGender(v, settings.gender));
    if (match) return { voice: match, genderMatched: true };
  }
  const sameGender = voices.find((v) => voiceMatchesGender(v, settings.gender));
  if (sameGender) return { voice: sameGender, genderMatched: true };
  const unknown = voices.find((v) => !voiceIsOtherGender(v, settings.gender));
  return { voice: unknown || voices[0] || null, genderMatched: false };
}

async function speak(text, speakerId) {
  if (!voiceEnabled) return;
  await voicesReady;
  const settings = VOICE_SETTINGS[speakerId] || DEFAULT_VOICE_SETTINGS;
  const utterance = new SpeechSynthesisUtterance(text);
  const { voice, genderMatched } = pickVoice(settings);
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
  }
  utterance.rate = 1.1;
  utterance.pitch = genderMatched ? settings.pitch : settings.pitch * MISMATCHED_VOICE_PITCH_FACTOR;
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

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (data.error) throw new Error(data.error);
  return data;
}

function showThinking(personaId) {
  const div = document.createElement("div");
  div.className = "message message-persona message-thinking";
  div.textContent = `${displayNames[personaId] || personaId} is thinking…`;
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return div;
}

// Requests advisors one at a time: each reply is shown and spoken while the next generates.
async function runRound(mode, message) {
  const advisors = Array.from(activePersonas);
  isBusy = true;
  updateActionButtons();
  try {
    if (message !== undefined) {
      await postJson("/chat/message", { session_id: sessionId, message });
    }
    for (const personaId of advisors) {
      const placeholder = showThinking(personaId);
      try {
        const data = await postJson("/chat/respond", {
          session_id: sessionId,
          persona_id: personaId,
          active_personas: advisors,
          mode,
        });
        placeholder.remove();
        renderTurn(data.turn);
      } catch (err) {
        placeholder.remove();
        throw err;
      }
    }
  } catch (err) {
    alert(err.message || "Request failed.");
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

  await runRound("reply", message);
});

debateButtonEl.addEventListener("click", async () => {
  if (isBusy || activePersonas.size < 2) return;
  await runRound("debate");
});

