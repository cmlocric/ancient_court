const sessionId = localStorage.getItem("session_id") || crypto.randomUUID();
localStorage.setItem("session_id", sessionId);

const transcriptEl = document.getElementById("transcript");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");
const personaChips = document.querySelectorAll(".persona-chip");

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

