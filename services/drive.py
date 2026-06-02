from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import HTTPException, status


class GoogleDriveService:
    async def apply_document_action(
        self,
        *,
        action: str,
        access_token: str | None,
        session_id: str,
        title: str,
        content: str,
        existing_document: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="La sesión de Google no incluye permiso de Drive/Docs. Inicia sesión de nuevo con los scopes ampliados.",
            )
        if action == "create":
            return await self.create_document(access_token, title, content, session_id)
        if action == "update":
            document_id = (existing_document or {}).get("document_id")
            if not document_id:
                return await self.create_document(access_token, title, content, session_id)
            return await self.update_document(access_token, document_id, title, content)
        if action == "delete":
            document_id = (existing_document or {}).get("document_id")
            if document_id:
                await self.delete_document(access_token, document_id)
            return None
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Acción de Drive no soportada.")

    async def create_document(self, access_token: str, title: str, content: str, session_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            create_response = await client.post(
                "https://docs.googleapis.com/v1/documents",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"title": title or f"NOVA {session_id}"},
            )
            if create_response.status_code >= 400:
                raise HTTPException(status_code=502, detail="Google Docs no permitió crear el documento.")
            document_id = create_response.json()["documentId"]
            await self._replace_document_body(client, access_token, document_id, content)
            await self._share_for_reading(client, access_token, document_id)
        return self._document_payload(document_id, shared=True)

    async def update_document(self, access_token: str, document_id: str, title: str, content: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            document_response = await client.get(
                f"https://docs.googleapis.com/v1/documents/{document_id}",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if document_response.status_code >= 400:
                raise HTTPException(status_code=502, detail="Google Docs no permitió leer el documento.")
            doc = document_response.json()
            end_index = max(1, int(doc.get("body", {}).get("content", [{}])[-1].get("endIndex", 1)) - 1)
            requests: list[dict[str, Any]] = []
            if end_index > 1:
                requests.append({"deleteContentRange": {"range": {"startIndex": 1, "endIndex": end_index}}})
            requests.append({"insertText": {"location": {"index": 1}, "text": content}})
            await client.post(
                f"https://docs.googleapis.com/v1/documents/{document_id}:batchUpdate",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"requests": requests},
            )
            if title:
                await client.patch(
                    f"https://www.googleapis.com/drive/v3/files/{document_id}",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={"name": title},
                )
        return self._document_payload(document_id, shared=True)

    async def delete_document(self, access_token: str, document_id: str) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(
                f"https://www.googleapis.com/drive/v3/files/{document_id}",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if response.status_code >= 400:
                raise HTTPException(status_code=502, detail="Google Drive no permitió borrar el documento.")

    async def _replace_document_body(
        self,
        client: httpx.AsyncClient,
        access_token: str,
        document_id: str,
        content: str,
    ) -> None:
        response = await client.post(
            f"https://docs.googleapis.com/v1/documents/{document_id}:batchUpdate",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"requests": [{"insertText": {"location": {"index": 1}, "text": content}}]},
        )
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail="Google Docs no permitió escribir el documento.")

    async def _share_for_reading(self, client: httpx.AsyncClient, access_token: str, document_id: str) -> None:
        await client.post(
            f"https://www.googleapis.com/drive/v3/files/{document_id}/permissions",
            params={"sendNotificationEmail": "false"},
            headers={"Authorization": f"Bearer {access_token}"},
            json={"role": "reader", "type": "anyone"},
        )

    def _document_payload(self, document_id: str, *, shared: bool) -> dict[str, Any]:
        return {
            "document_id": document_id,
            "url": f"https://docs.google.com/document/d/{document_id}/edit",
            "last_synced_at": datetime.now(timezone.utc).isoformat(),
            "shared": shared,
        }
