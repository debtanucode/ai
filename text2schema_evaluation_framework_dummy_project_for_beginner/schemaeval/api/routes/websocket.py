"""WebSocket endpoint for real-time evaluation streaming."""
from __future__ import annotations
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ...models.eval_request import EvalRequest
from ...engine.evaluator import EvaluationEngine
from ..dependencies import get_engine

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/evaluate")
async def ws_evaluate(websocket: WebSocket) -> None:
    engine: EvaluationEngine = get_engine()
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                request = EvalRequest.model_validate(data)
                await websocket.send_json({"type": "started", "run_id": request.run_id})
                result = await engine.evaluate(request)
                await websocket.send_json(
                    {"type": "result", "data": json.loads(result.full_result_json())}
                )
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
