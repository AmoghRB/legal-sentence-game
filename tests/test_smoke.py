"""Plumbing checks: the score store and the command line."""
import pytest

from app.storage import Store


def test_the_store_keeps_what_you_put_in_it(tmp_path):
    store = Store(str(tmp_path / "app.json"))
    store.put("games", [{"score": 25}])
    assert store.get("games") == [{"score": 25}]
    assert store.get("missing", "fallback") == "fallback"
    assert store.all() == {"games": [{"score": 25}]}


def test_the_command_line_lists_every_command(capsys):
    from app.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("play", "train", "classify", "predict", "evaluate", "history", "build-corpus"):
        assert command in out


def test_history_before_any_game(monkeypatch, tmp_path, capsys):
    from app.cli import main

    monkeypatch.setenv("DATA_PATH", str(tmp_path / "app.json"))
    main(["history"])
    assert "No games yet" in capsys.readouterr().out
