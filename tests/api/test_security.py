import pytest

OWN_ORIGIN = "http://127.0.0.1:8765"


@pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765", "localhost", "LOCALHOST:8765"])
def test_loopback_hosts_are_accepted(make_client, host):
    assert make_client().get("/api/health", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", ["testserver", "evil.example:8765", "127.0.0.1.evil.example",
                                  "192.168.1.5:8765", "[::1]:8765", ""])
def test_foreign_hosts_are_refused(make_client, host):
    """DNS rebinding sends the attacker's host name to the loopback address."""
    response = make_client().get("/api/health", headers={"host": host})
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid host header"}


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://localhost:5173",
                                    "http://127.0.0.1:8766"])
def test_foreign_origins_are_refused(make_client, method, origin):
    response = getattr(make_client(), method)("/api/health", headers={"origin": origin})
    assert response.status_code == 403


@pytest.mark.parametrize("origin", [OWN_ORIGIN, "http://localhost:8765"])
def test_own_origin_is_accepted(make_client, origin):
    assert make_client().get("/api/health", headers={"origin": origin}).status_code == 200


@pytest.mark.parametrize("site, status", [("same-origin", 200), ("none", 200), ("same-site", 403),
                                          ("cross-site", 403)])
def test_fetch_metadata_without_origin(make_client, site, status):
    """Media elements and navigations omit Origin but browsers still send Sec-Fetch-Site."""
    assert make_client().get("/api/health", headers={"sec-fetch-site": site}).status_code == status


def test_configured_development_origin_is_accepted(make_client):
    client = make_client(extra_origins=("http://localhost:5173",))
    assert client.get("/api/health", headers={"origin": "http://localhost:5173",
                                              "sec-fetch-site": "same-site"}).status_code == 200
    assert client.post("/api/health", headers={"origin": "http://localhost:5173"}).status_code == 405


def test_security_headers_on_accepted_and_refused_responses(make_client):
    client = make_client()
    for response in (client.get("/api/health"), client.get("/api/health", headers={"host": "evil"})):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["cross-origin-resource-policy"] == "same-origin"


def test_no_cross_origin_sharing_headers(make_client):
    response = make_client().options("/api/health", headers={
        "origin": OWN_ORIGIN, "access-control-request-method": "GET"})
    assert "access-control-allow-origin" not in response.headers
