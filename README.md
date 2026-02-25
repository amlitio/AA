# AA

## Autonomous Accounting Agent scaffold

This repository now includes a drop-in scaffold for an autonomous accounting system designed around Xero + document packet validation workflows.

### Added architecture layers

- `packages/agent/xerolens_agent`: policy-driven decision engine, risk scoring, and invoice-case workflow.
- `packages/doc_ai/xerolens_doc_ai`: extraction/matching/validation primitives for invoice + field-ticket packet processing.
- `packages/db/xerolens_db`: SQLAlchemy models, repository helpers, and initial Alembic migration for case/audit tables.
- `apps/api/xerolens_api`: API-level stubs for Firebase verification, Xero OAuth, webhook verification, documents, and cases routes.
- `apps/worker/xerolens_worker`: worker stubs for Celery scheduling, Xero sync hooks, and agent-run tasks.

### Existing training files

Legacy PyTorch experimentation files (`main.py`, `train.py`) remain present but are separate from the accounting agent scaffold.
