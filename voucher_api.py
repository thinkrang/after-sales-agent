"""Authenticated HTTP entry points for voucher disputes."""
from datetime import timezone
from typing import Annotated, Literal
import hashlib
import hmac
import json
import os
import time

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

import voucher_disputes as disputes
from auth import CurrentUser

router = APIRouter()
StoreCode = Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[A-Z0-9_-]+$")]


class RequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SubmitRequest(RequestBody):
    voucher_id: str = Field(min_length=1, max_length=64)
    claim: str = Field(min_length=1, max_length=500)
    store_code: StoreCode | None = None
    visit_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def require_store_for_visit(self):
        if self.visit_at is not None and self.store_code is None:
            raise ValueError("填写到店时间时需提供门店")
        return self


class ReviewVersion(RequestBody):
    expected_revision: int = Field(ge=1, strict=True)


class ReviewRequest(ReviewVersion):
    outcome: Literal["eligible_to_apply", "redemption_confirmed", "insufficient_evidence"]
    note: str = Field(min_length=1, max_length=500)


class MerchantEvent(RequestBody):
    event_id: str = Field(min_length=1, max_length=100)
    voucher_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z0-9_-]+$")
    source_version: int = Field(ge=1, le=2**63 - 1, strict=True)
    record_id: str = Field(min_length=1, max_length=100)
    status: Literal["unused", "redeemed"]
    redeem_ref: str | None = Field(default=None, min_length=1, max_length=100)
    redeemed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_redemption(self):
        if self.status == "redeemed":
            if not self.redeem_ref or self.redeemed_at is None:
                raise ValueError("已核销必须提供核销流水和时间")
            self.redeemed_at = self.redeemed_at.astimezone(timezone.utc)
        elif self.redeem_ref is not None or self.redeemed_at is not None:
            raise ValueError("未核销不能带核销流水或时间")
        return self


@router.get("/vouchers")
def list_owned_vouchers(user: CurrentUser):
    return {"vouchers": disputes.list_vouchers(user)}


@router.get("/vouchers/{voucher_id}/assessment")
def inspect_owned_voucher(voucher_id: str, user: CurrentUser):
    try:
        return disputes.inspect(user, voucher_id)
    except disputes.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/vouchers/{voucher_id}/visit-check")
def check_visit(voucher_id: str, user: CurrentUser, store_code: Annotated[StoreCode | None, Query()] = None,
                visit_at: AwareDatetime | None = None):
    try:
        return disputes.inspect_visit(user, voucher_id, store_code, visit_at)
    except disputes.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/voucher-disputes")
def list_disputes(user: CurrentUser, queue: bool = False):
    try:
        return {"cases": disputes.list_cases(user, queue=queue)}
    except disputes.Forbidden as exc:
        raise HTTPException(403, str(exc)) from exc


@router.post("/voucher-disputes")
def submit_dispute(body: SubmitRequest, user: CurrentUser):
    try:
        return disputes.submit(user, body.voucher_id, body.claim, store_code=body.store_code, visit_at=body.visit_at)
    except disputes.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except disputes.Conflict as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/voucher-disputes/{case_id}")
def dispute_detail(case_id: str, user: CurrentUser):
    try:
        return disputes.detail(user, case_id)
    except disputes.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc


def _review(user, case_id, revision, **kwargs):
    try:
        return disputes.review(user, case_id, revision, **kwargs)
    except disputes.Forbidden as exc:
        raise HTTPException(403, str(exc)) from exc
    except disputes.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except disputes.Conflict as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/voucher-disputes/{case_id}/recheck")
def refresh_evidence(case_id: str, body: ReviewVersion, user: CurrentUser):
    return _review(user, case_id, body.expected_revision)


@router.post("/voucher-disputes/{case_id}/decision")
def decide_dispute(case_id: str, body: ReviewRequest, user: CurrentUser):
    return _review(user, case_id, body.expected_revision, outcome=body.outcome, note=body.note)


@router.post("/voucher-evidence/{tenant}/merchant")
async def merchant_evidence(tenant: str, request: Request):
    """Receive a tenant-scoped, signed simulator event. Keys are opt-in."""
    from starlette.concurrency import run_in_threadpool

    try:
        keys = json.loads(os.environ.get("VOUCHER_MERCHANT_KEYS", "{}"))
        key = keys.get(tenant) if isinstance(keys, dict) else None
        stamp = request.headers.get("X-Voucher-Timestamp", "")
        signature = request.headers.get("X-Voucher-Signature", "").lower()
        if not isinstance(key, str) or not key or abs(int(time.time()) - int(stamp)) > 300:
            raise ValueError
    except ValueError as exc:
        raise HTTPException(401, "商家事件认证失败") from exc
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > 8192:
            raise HTTPException(413, "商家事件过大")
        raw.extend(chunk)
    expected = hmac.new(key.encode(), f"{tenant}.{stamp}.".encode() + raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected.encode(), signature.encode()):
        raise HTTPException(401, "商家事件认证失败")
    try:
        event = MerchantEvent.model_validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(422, "商家记录格式不合法") from exc
    try:
        payload = event.model_dump(mode="json", exclude={"event_id"})
        return await run_in_threadpool(disputes.ingest_merchant_evidence, tenant, event.event_id, payload)
    except disputes.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except disputes.Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
