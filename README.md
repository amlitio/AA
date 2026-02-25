# AA

## Autonomous Accounting Agent scaffold

This repository includes a scaffold for an autonomous accounting system designed around Xero + document packet validation workflows.

### Added architecture layers

- `packages/agent/xerolens_agent`: policy-driven decision engine, risk scoring, and invoice-case workflow.
- `packages/doc_ai/xerolens_doc_ai`: extraction/matching/validation primitives for invoice + field-ticket packet processing.
- `packages/db/xerolens_db`: SQLAlchemy models, repository helpers, and initial Alembic migration for case/audit tables.
- `apps/api/xerolens_api`: API-level stubs for Firebase verification, Xero OAuth, webhook verification, documents, and cases routes.
- `apps/worker/xerolens_worker`: worker stubs for Celery scheduling, Xero sync hooks, revenue reconciliation, and agent-run tasks.

### Revenue reconciliation defaults

- `InvoiceNumber` is treated as the field ticket number linkage key.
- Invoiced statuses include `DRAFT`, `SUBMITTED`, `AUTHORISED`, and `PAID`.
- `apps/worker/xerolens_worker/tasks/reconcile_revenue.py` computes:
  - `REVENUE_LEAKAGE` (ticket exists, invoice missing)
  - `UNSUPPORTED_BILLING` (invoice exists, packet missing)
  - `DUPLICATE` (same invoice number appears more than once)
  - `AMOUNT_MISMATCH` (packet total vs invoice total outside tolerance)

### Existing training files

Legacy PyTorch experimentation files (`main.py`, `train.py`) remain present but are separate from the accounting agent scaffold.
