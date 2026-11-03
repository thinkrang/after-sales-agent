# Security model

This project is a demo of an Agent that can execute a consequential refund
action. It treats the model and retrieved documents as untrusted inputs.

## Trust boundaries

| Component | Trust level | Enforcement |
|---|---|---|
| Browser request | untrusted | Bearer token, Pydantic validation |
| LLM tool arguments | untrusted | order ownership, policy and approval checks |
| Retrieved documents | untrusted | tenant-filtered retrieval and evidence gate |
| Approval record | server-owned | actor, tenant, thread, order, amount and payload hash binding |
| Risk tier | server-owned | deterministic rules in `risk.py`; the model cannot influence the tier |
| Refund gateway | external | idempotency key and database uniqueness |

## Implemented controls

- Passwords use salted PBKDF2 hashes; access tokens are signed and expire.
- Tenant and roles come from the server-issued token, not request bodies.
- Session, order, ticket and knowledge-document reads are scoped to the
  authenticated owner; unauthorized session ids return `404`.
- A refund cannot execute without an `approved` ticket matching the exact
  tenant, requester, thread, order, amount and normalized reason.
- Risk tier is computed server-side by pure functions (amount / window edge /
  recent frequency). High-risk tickets require an `approver` role and reject
  self-approval, so the requester cannot clear their own risky refund.
- The risk rule version is stored on the ticket, so an auditor can tell which
  rule set approved a given refund.
- Approved -> executing is an atomic state transition checked by `rowcount`.
- The downstream refund uses the ticket id as an idempotency key. A unique
  database constraint prevents two successful refunds for one execution key.
- A retry after a downstream success but before ticket completion reuses the
  same key and reconciles the original refund id.
- Refund task edits use an optimistic revision. Changing/cancelling the task
  and rejecting its old pending approvals share a SQLite write transaction.
  If approval wins the race, editing is rejected; if editing wins, the old
  approval cannot execute. Already executed refunds cannot be withdrawn.
- New chat actions require a registered task with apply intent and matching
  revision/order/reason. Existing legacy pending approvals remain resumable.
  Task intent is model-extracted, not an authorization grant; human approval
  and the server's execution checks remain mandatory.
- New conversation tools only submit/withdraw proposals. Approval/execution
  runs in a separate checkpointed graph. Resuming that graph without a real
  server-owned decision cannot authorize a refund. Dialogue stays runnable
  while approval waits; old dialogue interrupts remain supported for upgrades.
- In-process per-conversation guards protect dialogue checkpoint writes from
  simultaneous chat/result updates. They do not provide distributed locking
  across multiple API workers; database CAS and uniqueness still guard money.

## Explicit demo limitations

- Demo users and `demo1234` passwords are seeded into SQLite. Production
  deployments must set `AUTH_SEED_DEMO_USERS=0` and use an OIDC/enterprise
  identity provider.
- The default token signing secret is for local demonstration only. Set a
  random `AUTH_SECRET` outside the repository.
- The payment/order service is simulated locally. Production integration needs
  mutual authentication, provider-side idempotency verification, reconciliation
  jobs and an operator compensation UI.
- The browser demo stores the Bearer token in `localStorage`. A production web
  frontend should prefer an HttpOnly/SameSite cookie or a backend-for-frontend
  session with CSRF protection.
- Low-risk HITL is customer consent. High-risk tickets require a separate
  approver in the same tenant and prohibit self-approval. Requesters may
  withdraw a still-pending high-risk task through the task endpoint.
- Rate limiting, WAF, secret rotation and SIEM export are deployment concerns
  not implemented in this repository.

## High-risk regressions

The security test suite should always cover:

1. reading another tenant's session;
2. changing `tenant` in a chat request;
3. approving an order that belongs to another user;
4. reusing an approval for a different order, amount, thread or reason;
4b. self-approving a high-risk refund, or a non-approver clearing one;
5. double-clicking approval and concurrent execution;
6. crashing after downstream success but before marking the ticket executed.
