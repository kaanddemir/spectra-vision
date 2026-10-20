import json
import tomllib

from spectra import __version__
from spectra.api import API_VERSION
from spectra.api.settings import REPOSITORY_ROOT


def test_product_version_has_one_source():
    assert __version__ == "2.0.0"
    assert API_VERSION == __version__
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    assert "version" not in project["project"] and "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "spectra.__version__"}


def test_web_application_matches_product_version():
    package = json.loads((REPOSITORY_ROOT / "frontend" / "package.json").read_text())
    lock = json.loads((REPOSITORY_ROOT / "frontend" / "package-lock.json").read_text())
    assert package["version"] == lock["version"] == lock["packages"][""]["version"] == __version__


def test_health_and_openapi_report_product_version(make_client):
    client = make_client()
    assert client.get("/api/health").json()["version"] == "2.0.0"
    assert client.get("/api/openapi.json").json()["info"]["version"] == "2.0.0"
