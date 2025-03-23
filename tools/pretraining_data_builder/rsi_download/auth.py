import httpx
import msgspec


class CDSETokens(msgspec.Struct):
    """Copernicus Data Space Ecosystem Tokens"""

    access_token: str
    refresh_token: str
    expires_in: int
    refresh_expires_in: int
    token_type: str
    not_before_policy: int = msgspec.field(name="not-before-policy")
    session_state: str
    scope: str


def get_access_token(username: str, password: str) -> CDSETokens:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        with httpx.Client() as client:
            r = client.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data=data,
            )
            r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Access token creation failed: {e}. Reponse from the server was: {r.json()}"
        )
    return msgspec.json.decode(r.content, type=CDSETokens)
