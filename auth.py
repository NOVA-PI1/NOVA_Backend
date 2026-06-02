import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import HTTPException, Request, status
from fastapi.responses import RedirectResponse

from config import Settings


GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_OAUTH_SCOPES = "openid email profile https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/documents"


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def create_jwt(payload: dict[str, Any], secret_key: str, expires_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        **payload,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = ".".join(
        [
            _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _b64url_encode(json.dumps(claims, separators=(",", ":"), default=str).encode("utf-8")),
        ]
    )
    signature = hmac.new(secret_key.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def decode_jwt(token: str, secret_key: str) -> dict[str, Any]:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
        signing_input = f"{header_segment}.{payload_segment}"
        expected_signature = hmac.new(
            secret_key.encode("utf-8"),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        received_signature = _b64url_decode(signature_segment)
        if not hmac.compare_digest(expected_signature, received_signature):
            raise ValueError("invalid signature")

        payload = json.loads(_b64url_decode(payload_segment))
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticación inválido.",
        ) from error

    expires_at = payload.get("exp")
    if not isinstance(expires_at, int) or expires_at < int(datetime.now(timezone.utc).timestamp()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La sesión expiró. Inicia sesión de nuevo.",
        )

    return payload


def configured_providers(settings: Settings) -> list[str]:
    providers: list[str] = []
    if settings.google_client_id and settings.google_client_secret:
        providers.append("google")
    return providers


def get_bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def get_current_user_from_request(request: Request, settings: Settings) -> dict[str, Any]:
    token = get_bearer_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No se recibió token de autenticación.",
        )
    payload = decode_jwt(token, settings.secret_key)
    user = payload.get("user")
    if not isinstance(user, dict):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token sin perfil de usuario.")
    return user


def build_redirect_uri(request: Request, settings: Settings) -> str:
    if settings.google_redirect_uri:
        return settings.google_redirect_uri
    return str(request.url_for("google_callback"))


def create_oauth_state(frontend_redirect_url: str, settings: Settings) -> str:
    return create_jwt(
        {"frontend_redirect_url": frontend_redirect_url, "purpose": "oauth_state"},
        settings.secret_key,
        expires_minutes=10,
    )


def read_oauth_state(state: str, settings: Settings) -> str:
    payload = decode_jwt(state, settings.secret_key)
    if payload.get("purpose") != "oauth_state":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Estado OAuth inválido.")
    frontend_redirect_url = payload.get("frontend_redirect_url")
    if not isinstance(frontend_redirect_url, str):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Redirección frontend inválida.")
    return frontend_redirect_url


def is_allowed_frontend_redirect(url: str, settings: Settings) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    if parsed.hostname in {"localhost", "127.0.0.1"}:
        return True

    allowed_hosts = {urlparse(settings.frontend_base_url).hostname}
    for origin in settings.cors_allowed_origins.split(","):
        origin = origin.strip()
        if not origin or origin == "*":
            return True
        origin_host = urlparse(origin if "://" in origin else f"https://{origin}").hostname
        allowed_hosts.add(origin_host)
    return parsed.hostname in allowed_hosts


def redirect_with_auth_error(frontend_redirect_url: str, error: str) -> RedirectResponse:
    query = urlencode({"auth_error": error})
    separator = "&" if "?" in frontend_redirect_url else "?"
    return RedirectResponse(f"{frontend_redirect_url}{separator}{query}", status_code=302)


async def exchange_google_code(code: str, request: Request, settings: Settings) -> dict[str, Any]:
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Google OAuth no está configurado.")

    redirect_uri = build_redirect_uri(request, settings)
    async with httpx.AsyncClient(timeout=20) as client:
        token_response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
        )
        if token_response.status_code >= 400:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google rechazó el código OAuth.")

        access_token = token_response.json().get("access_token")
        if not access_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google no devolvió access_token.")

        userinfo_response = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        if userinfo_response.status_code >= 400:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No se pudo leer el perfil de Google.")

    profile = userinfo_response.json()
    google_id = profile.get("sub")
    email = profile.get("email")
    if not google_id or not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Perfil de Google incompleto.")

    return {
        "user_id": f"google:{google_id}",
        "provider": "google",
        "provider_user_id": google_id,
        "email": email,
        "name": profile.get("name") or email,
        "avatar_url": profile.get("picture"),
        "email_verified": bool(profile.get("email_verified")),
        "google_access_token": access_token,
    }


def create_google_authorize_redirect(
    request: Request,
    settings: Settings,
    frontend_redirect_url: str,
) -> RedirectResponse:
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Google OAuth no está configurado.")
    if not is_allowed_frontend_redirect(frontend_redirect_url, settings):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Frontend redirect no permitido.")

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": build_redirect_uri(request, settings),
        "response_type": "code",
        "scope": GOOGLE_OAUTH_SCOPES,
        "access_type": "offline",
        "prompt": "select_account",
        "state": create_oauth_state(frontend_redirect_url, settings),
    }
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}", status_code=302)
