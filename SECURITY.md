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
| Refund gateway | external | idempotency key and database uniqueness |

## Implemented controls

- Passwords use salted PBKDF2 hashes; access tokens are signed and expire.
- Tenant and roles come from the server-issued token, not request bodies.
- Session, order, ticket and knowledge-document reads are scoped to the
  authenticated owner; unauthorized session ids return `404`.
- A refund cannot execute without an `approved` ticket matching the exact
  tenant, requester, thread, order, amount and normalized reason.
- Approved -> executing is an atomic state transition checked by `rowcount`.
- The downstream refund uses the ticket id as an idempotency key. A unique
  database constraint prevents two successful refunds for one execution key.
- A retry after a downstream success but before ticket completion reuses the
  same key and reconciles the original refund id.

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
- The current HITL record is the authenticated customer's consent to execute
  the proposed action. If business policy requires four-eyes review, add a
  separate `approver` actor and reject self-approval in the workflow service.
- Rate limiting, WAF, secret rotation and SIEM export are deployment concerns
  not implemented in this repository.

## High-risk regressions

The security test suite should always cover:

1. reading another tenant's session;
2. changing `tenant` in a chat request;
3. approving an order that belongs to another user;
4. reusing an approval for a different order, amount, thread or reason;
5. double-clicking approval and concurrent execution;
6. crashing after downstream success but before marking the ticket executed.
