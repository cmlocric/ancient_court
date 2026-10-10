const sessionId = localStorage.getItem("session_id") || crypto.randomUUID();
localStorage.setItem("session_id", sessionId);

const transcriptEl = document.getElementById("transcript");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");

function renderTurn(turn) {
  const div = document.createElement("div");
  div.className = "message";
  const speaker = document.createElement("span");
  speaker.className = "speaker";
  speaker.textContent = turn.speaker;
  const text = document.createElement("span");
  text.textContent = turn.text;
  div.appendChild(speaker);
  div.appendChild(text);
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;

  const activePersonas = Array.from(document.querySelectorAll(".persona-checkbox:checked")).map(
    (el) => el.value
  );

  inputEl.value = "";
  renderTurn({ speaker: "user", text: message });

  const response = await fetch("/chat/send", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message, active_personas: activePersonas }),
  });

  const data = await response.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  // The user's own turn was already rendered immediately above; only render replies.
  data.turns.filter((turn) => turn.speaker !== "user").forEach(renderTurn);
});
