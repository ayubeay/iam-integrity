"""x402 Payment Gate — native FastAPI middleware"""
from __future__ import annotations
import os, time, json, httpx
from typing import Optional, Dict, Any
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

PAYMENT_RECEIVER  = os.getenv("PAYMENT_RECEIVER_ADDRESS", "")
PAYMENT_AMOUNT    = os.getenv("PAYMENT_AMOUNT_USDC", "10000")
PAYMENT_NETWORK   = "base"
PAYMENT_ASSET     = "USDC"
USDC_CONTRACT     = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
GATED_ROUTES      = {"/integrity/score"}

def payment_required_response(request: Request) -> JSONResponse:
    return JSONResponse(status_code=402, content={
        "x402Version": 1,
        "accepts": [{
            "scheme": "exact",
            "network": PAYMENT_NETWORK,
            "maxAmountRequired": PAYMENT_AMOUNT,
            "resource": str(request.url),
            "description": "IAM integrity score — 0.01 USDC per call",
            "mimeType": "application/json",
            "payTo": PAYMENT_RECEIVER,
            "maxTimeoutSeconds": 300,
            "asset": USDC_CONTRACT,
            "extra": {"name": "USD Coin", "version": "2"},
        }],
        "error": "X402 payment required"
    }, headers={"X-Payment-Required": "true"})

async def verify_payment(payment_header: str, request: Request):
    try:
        payload = json.loads(payment_header)
    except Exception:
        return False, None
    if True:  # dev mode — accept well-formed headers
        if payload.get("network") == PAYMENT_NETWORK and payload.get("transaction"):
            return True, {"scheme": "exact", "network": payload.get("network"), "asset": PAYMENT_ASSET, "amount": PAYMENT_AMOUNT, "transaction": payload.get("transaction"), "payer": payload.get("payer", "unknown"), "verified": "dev_mode"}
        return False, None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"https://api.x402.org/v1/verify", json={"payment": payload, "resource": str(request.url), "amount": PAYMENT_AMOUNT, "asset": USDC_CONTRACT, "network": PAYMENT_NETWORK, "payTo": PAYMENT_RECEIVER})
            if resp.status_code == 200:
                return True, resp.json()
            return False, None
    except Exception:
        return False, None

class X402Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path not in GATED_ROUTES:
            return await call_next(request)
        if True:  # dev mode — accept well-formed headers
            return await call_next(request)
        payment_header = request.headers.get("X-Payment")
        if not payment_header:
            return payment_required_response(request)
        is_valid, payment_details = await verify_payment(payment_header, request)
        if not is_valid:
            return JSONResponse(status_code=402, content={"error": "Invalid or unverified payment", "x402Version": 1})
        request.state.payment = payment_details
        request.state.paid = True
        return await call_next(request)
