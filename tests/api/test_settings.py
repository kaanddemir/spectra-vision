import stat
import subprocess

import pytest

from spectra.api.settings import REPOSITORY_ROOT, Settings, SettingsError


def test_defaults_use_ignored_repository_data_directory():
    settings = Settings()
    assert settings.data_dir == REPOSITORY_ROOT / "var"
    assert settings.models_dir == REPOSITORY_ROOT / "models" / "engine"
    assert settings.database_path.parent == settings.data_dir
    assert settings.allowed_origins == {"http://127.0.0.1:8765", "http://localhost:8765"}
    ignored = subprocess.run(["git", "check-ignore", "--quiet", "var/spectra.sqlite3"], cwd=REPOSITORY_ROOT)
    assert ignored.returncode == 0


def test_prepare_creates_private_directories(tmp_path):
    settings = Settings(data_dir=tmp_path / "state")
    settings.prepare()
    settings.prepare()
    for path in (settings.data_dir, settings.cache_dir, settings.evidence_dir, settings.tmp_dir):
        assert path.is_dir()
    assert stat.S_IMODE(settings.data_dir.stat().st_mode) == 0o700


def test_library_folders_are_resolved_and_deduplicated(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    (tmp_path / "alias").symlink_to(folder)
    settings = Settings(data_dir=tmp_path / "state", library_dirs=(folder, tmp_path / "alias", str(folder)))
    assert settings.library_dirs == (folder.resolve(),)


@pytest.mark.parametrize("problem", ["missing", "file", "inside_data"])
def test_invalid_library_folders_are_refused(tmp_path, problem):
    data = tmp_path / "state"
    target = {"missing": tmp_path / "absent", "file": tmp_path / "video.mp4",
              "inside_data": data / "cache"}[problem]
    if problem == "file":
        target.write_bytes(b"")
    if problem == "inside_data":
        target.mkdir(parents=True)
    with pytest.raises(SettingsError):
        Settings(data_dir=data, library_dirs=(target,))


def test_data_directory_may_sit_inside_a_library_folder(tmp_path):
    settings = Settings(data_dir=tmp_path / "state", library_dirs=(tmp_path,))
    assert settings.library_dirs == (tmp_path.resolve(),)


@pytest.mark.parametrize("origin", ["https://localhost:5173", "http://evil.example:5173", "http://localhost",
                                    "http://localhost:5173/app", "http://user@localhost:5173", "localhost:5173",
                                    "http://localhost:99999"])
def test_extra_origins_must_be_loopback_http_origins(origin):
    with pytest.raises(SettingsError):
        Settings(extra_origins=(origin,))


def test_extra_origin_is_normalized():
    settings = Settings(extra_origins=("http://LOCALHOST:5173",))
    assert "http://localhost:5173" in settings.allowed_origins


@pytest.mark.parametrize("port", [0, 65536, True, "8765"])
def test_invalid_ports_are_refused(port):
    with pytest.raises(SettingsError):
        Settings(port=port)


def test_command_line_serves_only_on_loopback(tmp_path, monkeypatch):
    import uvicorn

    from spectra.api.__main__ import main
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: calls.append((app, options)))
    main(["--port", "9001", "--data-dir", str(tmp_path / "state"), "--library", str(tmp_path),
          "--allow-origin", "http://localhost:5173"])
    app, options = calls[0]
    assert options["host"] == "127.0.0.1"
    assert options["port"] == 9001
    assert options["proxy_headers"] is False
    assert options["timeout_graceful_shutdown"] == 3  # a followed analysis cannot keep a stopping server alive
    assert options["log_level"] == "info"
    main(["--port", "9002", "--data-dir", str(tmp_path / "state"), "--log-level", "warning"])
    assert calls[1][1]["log_level"] == "warning"
    assert app.state.settings.library_dirs == (tmp_path.resolve(),)
    assert "http://localhost:5173" in app.state.settings.allowed_origins


def test_command_line_rejects_invalid_settings_and_host_option(tmp_path, capsys):
    from spectra.api.__main__ import parse_settings
    with pytest.raises(SystemExit):
        parse_settings(["--library", str(tmp_path / "absent")])
    assert "library folder does not exist" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parse_settings(["--host", "0.0.0.0"])
