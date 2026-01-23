from __future__ import annotations

from typing import cast

from fastapi import HTTPException, Request

from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool


def get_server(request: Request) -> AsyncVoxCPMServerPool:
    server = getattr(request.app.state, "server", None)
    if server is None:
        raise HTTPException(status_code=503, detail="Model server not ready")
    # app.state is dynamically typed; normalize for type checkers.
    return cast(AsyncVoxCPMServerPool, server)
