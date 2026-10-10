import json

from src.persona import registry
from src.persona.dataset_builder import build_dataset
from src.persona.registry import PersonaConfig


def _write_fixture_json(path, num_sentences: int) -> None:
    sentences = [
        f"This is reflection number {i} about virtue, patience, and the pursuit of a good life."
        for i in range(num_sentences)
    ]
    payload = {
        "source": str(path),
        "pages": [{"page": 1, "text": " ".join(sentences)}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_dataset_produces_valid_split_jsonl(tmp_path, monkeypatch):
    source_json = tmp_path / "fake_persona.json"
    _write_fixture_json(source_json, num_sentences=60)

    fake_persona = PersonaConfig(
        persona_id="fake_persona",
        display_name="Fake Persona",
        title="Test Persona",
        source_json=source_json,
        system_prompt="You are a fake persona used only for testing.",
    )
    monkeypatch.setitem(registry.PERSONAS, "fake_persona", fake_persona)

    train_path, val_path = build_dataset("fake_persona", val_split=0.1, output_dir=tmp_path)

    assert train_path.exists()
    assert val_path.exists()

    train_records = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    val_records = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines()]

    total = len(train_records) + len(val_records)
    assert total >= 5
    assert len(train_records) == int(total * 0.9)

    for record in train_records + val_records:
        assert record["system"] == fake_persona.system_prompt
        assert isinstance(record["instruction"], str) and record["instruction"]
        assert isinstance(record["response"], str) and record["response"]
