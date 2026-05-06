from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/igc", tags=["igc"])


@router.post("/analyze")
async def analyze_igc():
    raise HTTPException(status_code=501, detail="IGC analysis is not yet implemented.")
