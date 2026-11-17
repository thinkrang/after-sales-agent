"""Launch a loopback-only V200 browser demo with a fresh temporary database."""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V200 核销争议演示</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;max-width:650px;margin:36px auto;padding:0 16px}
button{font:inherit;padding:6px 12px;margin:4px} output{display:block;padding:8px 0}
</style>
<h1>V200 核销争议演示</h1>
<p>先以消费者身份提交争议，再以主管身份打开待复核单。保持详情页打开，从这里补传商家流水。</p>
<button data-user="customer">以消费者身份打开</button>
<button data-user="reviewer">以主管身份打开</button>
<p><button data-event="late">补传商家流水</button>
<button data-event="old">发送旧版本</button></p>
<output aria-live="polite"></output>
<script>
async function post(url, data) {
  const res = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(data)});
  const body = await res.json();
  if (!res.ok) throw new Error(body.detail || "操作失败");
  return body;
}
const messages = {applied:"流水已接收", duplicate:"重复事件已忽略", ignored_old_version:"旧版本已忽略"};
for (const button of document.querySelectorAll("button")) {
  button.onclick = async () => {
    button.disabled = true;
    const status = document.querySelector("output");
    try {
      if (button.dataset.user) {
        const data = await post("/demo/login", {identity:button.dataset.user});
        localStorage.setItem("after_sales_access_token", data.access_token);
        location.href = "/voucher-center?voucher=V200&store=CENTRAL";
      } else {
        const data = await post("/demo/evidence", {event:button.dataset.event});
        status.textContent = messages[data.status] + (data.case_rechecked ? "，V200 已重新核查" : "");
      }
    } catch (error) { status.textContent = error.message; }
    finally { button.disabled = false; }
  };
}
</script></html>"""


def build_app(port, key, *, mixed=False):
    from typing import Literal

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel

    import voucher_disputes as disputes
    from auth import CurrentUser, User, create_access_token
    from voucher_api import router

    app = FastAPI()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1"])
    app.include_router(router)
    customer = User("acme-user", "演示消费者", "acme", frozenset({"customer"}))
    reviewer = User("acme-manager", "演示主管", "acme", frozenset({"approver"}))

    class DemoLogin(BaseModel):
        identity: Literal["customer", "reviewer"]

    class DemoEvidence(BaseModel):
        event: Literal["late", "old"]

    @app.get("/", response_class=HTMLResponse)
    def home():
        if mixed:
            return FileResponse(ROOT / "scripts" / "mixed_demo.html")
        return PAGE

    @app.get("/voucher-center")
    def voucher_center():
        return FileResponse(ROOT / "voucher_center.html")

    @app.get("/auth/me")
    def current_user(user: CurrentUser):
        return {"username": user.username, "display_name": user.display_name, "tenant_id": user.tenant_id,
                "roles": sorted(user.roles)}

    @app.post("/demo/login")
    def demo_login(body: DemoLogin):
        return {"access_token": create_access_token(customer if body.identity == "customer" else reviewer)}

    @app.post("/demo/evidence")
    def demo_evidence(body: DemoEvidence):
        records = disputes.inspect(customer, "V200")["evidence"]
        platform = next(row for row in records if row["source"] == "platform")
        if body.event == "late":
            event = {"event_id": "demo-v200-new", "voucher_id": "V200", "source_version": 2,
                     "record_id": "pos-v200", "status": "redeemed", "redeem_ref": platform["redeem_ref"],
                     "redeemed_at": platform["redeemed_at"]}
        else:
            event = {"event_id": "demo-v200-old", "voucher_id": "V200", "source_version": 1,
                     "record_id": "pos-v200-old", "status": "unused"}
        raw = json.dumps(event, separators=(",", ":")).encode("utf-8")
        stamp = str(int(time.time()))
        signature = hmac.new(key.encode(), f"acme.{stamp}.".encode() + raw, hashlib.sha256).hexdigest()
        req = Request(f"http://127.0.0.1:{port}/voucher-evidence/acme/merchant", data=raw, headers={
            "Content-Type": "application/json", "X-Voucher-Timestamp": stamp, "X-Voucher-Signature": signature,
        })
        try:
            with build_opener(ProxyHandler({})).open(req, timeout=5) as response:
                return json.load(response)
        except HTTPError as exc:
            raise HTTPException(exc.code, "演示回调被拒绝") from exc
        except OSError as exc:
            raise HTTPException(502, "无法连接本地演示回调") from exc

    if mixed:
        from scripts.mixed_demo import install
        install(app)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mixed", action="store_true", help="Run the offline mixed-task browser replay")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    import uvicorn

    with tempfile.TemporaryDirectory(prefix="voucher-v200-demo-") as folder:
        key = secrets.token_urlsafe(32)
        os.environ.update(AGENT_DATA_DIR=folder, AUTH_SECRET=secrets.token_urlsafe(32),
                          VOUCHER_MERCHANT_KEYS=json.dumps({"acme": key}))
        if args.mixed:
            os.environ.update(LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false")
        import db
        import tickets
        import voucher_disputes as disputes

        try:
            tickets.init_tickets_table()
            disputes.seed_demo()
            if args.mixed:
                import orders
                orders.init_order_db()
            app = build_app(args.port, key, mixed=args.mixed)
            uvicorn.run(app, host="127.0.0.1", port=args.port)
        finally:
            db.release_all()


if __name__ == "__main__":
    main()
