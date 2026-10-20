"""Serving the built web application next to the API."""
import pytest

from spectra.api.frontend import CONTENT_SECURITY_POLICY


@pytest.fixture
def build(tmp_path):
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>Spectra</title>")
    (root / "assets" / "index-abc123.js").write_text("console.log('app');")
    (root / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    (root / ".env").write_text("SECRET=1")
    (tmp_path / "secret.txt").write_text("outside the build")
    return root


def test_without_a_build_only_the_api_is_served(make_client):
    client = make_client()
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 404
    assert client.get("/videos").status_code == 404


def test_a_directory_without_index_is_not_served(make_client, tmp_path):
    (tmp_path / "empty").mkdir()
    assert make_client(frontend_dir=tmp_path / "empty").get("/").status_code == 404


@pytest.mark.parametrize("path", ["/", "/videos", "/videos/0123456789abcdef0123456789abcdef", "/library"])
def test_application_routes_return_the_page_with_a_strict_policy(make_client, build, path):
    response = make_client(frontend_dir=build).get(path)
    assert response.status_code == 200
    assert response.text.startswith("<!doctype html>")
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert "'unsafe-inline'" not in CONTENT_SECURITY_POLICY
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-frame-options"] == "DENY"
    assert make_client(frontend_dir=build).head(path).status_code == 200


def test_build_files_are_served_with_their_types(make_client, build):
    client = make_client(frontend_dir=build)
    script = client.get("/assets/index-abc123.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert "immutable" in script.headers["cache-control"]
    icon = client.get("/favicon.svg")
    assert icon.headers["content-type"].startswith("image/svg+xml")
    assert icon.headers["cache-control"] == "no-cache"


@pytest.mark.parametrize("path", ["/assets/missing.js", "/missing.css"])
def test_missing_files_are_not_answered_with_the_page(make_client, build, path):
    assert make_client(frontend_dir=build).get(path).status_code == 404


@pytest.mark.parametrize("path", ["/.env", "/assets/../../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt"])
def test_hidden_files_and_paths_outside_the_build_are_never_served(make_client, build, path):
    response = make_client(frontend_dir=build).get(path)
    assert "SECRET" not in response.text
    assert "outside the build" not in response.text


def test_api_paths_never_fall_back_to_the_page(make_client, build):
    client = make_client(frontend_dir=build)
    assert client.get("/api/health").json()["status"] in ("ok", "degraded")
    missing = client.get("/api/unknown")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Not Found"}
    assert client.get("/api").status_code == 404


def test_page_requests_pass_the_same_host_and_origin_checks(make_client, build):
    client = make_client(frontend_dir=build)
    assert client.get("/", headers={"host": "evil.example:8765"}).status_code == 400
    assert client.get("/", headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert client.get("/", headers={"sec-fetch-site": "none"}).status_code == 200


def test_command_line_serves_a_named_build_and_rejects_a_missing_one(build, tmp_path, capsys):
    from spectra.api.__main__ import parse_settings
    assert parse_settings(["--frontend", str(build)]).frontend_dir == build.resolve()
    with pytest.raises(SystemExit):
        parse_settings(["--frontend", str(tmp_path / "absent")])
    assert "no web application build" in capsys.readouterr().err
