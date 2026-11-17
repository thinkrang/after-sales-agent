"""Pure, versioned voucher rules shared by online assessment and offline replay."""
from datetime import datetime, timedelta

POLICIES = {
    "voucher-v1": timedelta(minutes=5),
    "voucher-v2": timedelta(minutes=2),
}
ACTIVE_POLICY = "voucher-v2"


def parse_aware_datetime(value: str | None) -> datetime:
    parsed = datetime.fromisoformat(value or "")
    if parsed.tzinfo is None:
        raise ValueError("missing timezone")
    return parsed


def evaluate_redemption(voucher: dict, evidence: list[dict], now: datetime) -> tuple[str, str]:
    tolerance = POLICIES.get(voucher["policy_version"])
    if tolerance is None:
        return "manual_review", "unknown_policy"
    sources = {record["source"]: record for record in evidence}
    if {"platform", "merchant"} - sources.keys():
        return "manual_review", "missing_source"
    platform, merchant = sources["platform"], sources["merchant"]
    if platform["status"] != merchant["status"]:
        return "manual_review", "status_mismatch"
    try:
        expiry = parse_aware_datetime(voucher["expires_at"])
        if platform["status"] == "unused":
            if any(row["redeem_ref"] or row["redeemed_at"] for row in (platform, merchant)):
                return "manual_review", "inconsistent_record"
            if now >= expiry:
                return "manual_review", "expired_voucher"
            return "eligible_to_apply", "unused_and_valid"
        if platform["status"] == "redeemed":
            if not platform["redeem_ref"] or platform["redeem_ref"] != merchant["redeem_ref"]:
                return "manual_review", "redemption_reference_mismatch"
            first = parse_aware_datetime(platform["redeemed_at"])
            second = parse_aware_datetime(merchant["redeemed_at"])
            if max(first, second) > min(expiry, now) or abs(first - second) > tolerance:
                return "manual_review", "redemption_time_mismatch"
            return "redemption_supported", "matching_redemption"
    except ValueError:
        return "manual_review", "invalid_timestamp"
    return "manual_review", "unexpected_status"


def initial_dispute_result(recommendation: str, *, on_site: bool) -> tuple[str, str | None]:
    if recommendation == "eligible_to_apply" and not on_site:
        return "resolved", "eligible_to_apply"
    return "pending_review", None
