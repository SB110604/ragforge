# SmartStock RPA — Architecture

SmartStock RPA is organised in the five layers used in the project report, with a continuous-learning loop feeding
the AI layer back from the automation and output layers.

```mermaid
flowchart TB
  subgraph L1["1 · Data input"]
    F[Forms & tables]:::in
    IMP[CSV / Excel import<br/>map → preview → commit]:::in
    SCAN[Browser barcode / QR scan<br/>html5-qrcode]:::in
    OCR[Invoice OCR<br/>Tesseract + RapidFuzz]:::in
    POS[POS sales API<br/>X-API-Key]:::in
    MAIL[Email quotation reader<br/>IMAP, optional]:::in
  end
  subgraph L2["2 · Processing & storage"]
    API[FastAPI · Pydantic validation<br/>permissions per route]
    SVC[Services: stock ledger + FEFO,<br/>sales, GST, PDFs, audit]
    PG[(PostgreSQL 16<br/>Alembic migrations)]
  end
  subgraph L3["3 · Agentic AI"]
    ORCH{{Orchestrator}}
    A1[Stock Monitor] --> A2[Expiry] --> A3[Decision<br/>forecast · SS · ROP · EOQ] --> A4[Supplier] --> A5[Auto-order] --> A6[Tracking] --> A7[Anomaly]
    LRN[Continuous learning<br/>accuracy · drift · retrain]
    CHAT[Chat assistant<br/>tools + LLM / rules]
    ML[(Model registry<br/>joblib files)]
  end
  subgraph L4["4 · Automation (RPA engine)"]
    WF[app/workflow state machines<br/>PR · PO · transfer · return · RFQ]
    APR[Approval engine<br/>rules · levels · tokens · SLA]
    BEAT[Celery Beat + workers<br/>schedules, emails]
  end
  subgraph L5["5 · Output"]
    UI[React PWA dashboards<br/>en / hi / mr, dark mode]
    NOTIF[In-app · email · Telegram]
    DOCS[PO / GRN / debit-note PDFs<br/>reports PDF + Excel]
  end
  L1 --> API --> SVC --> PG
  PG --> ORCH --> A1
  A5 --> APR --> WF
  WF --> NOTIF & DOCS
  WF --> PG
  BEAT --> ORCH & APR & WF
  PG --> UI
  UI -- approve / edit / reject --> APR
  APR -- feedback --> LRN
  PG -- new sales --> LRN
  LRN -- correction factor, promoted model --> A3
  A3 <--> ML
  CHAT -. read-only tools .-> PG
```

## Layer by layer

| Layer | What happens | Main code |
|---|---|---|
| 1 Data input | Forms with zod + Pydantic validation; CSV/XLSX import with auto column mapping, per-row errors and error report; barcode/QR scanning (camera, USB scanner or typed); OCR of supplier bills into a pre-filled invoice; POS sales API; IMAP price-update reader | `frontend/src/pages/*`, `app/services/imports.py`, `app/services/ocr.py`, `app/api/v1/operations.py`, `app/services/quote_reader.py` |
| 2 Processing & storage | REST API under `/api/v1`, permission dependency on every route, service layer writes the stock ledger (batch + signed movement), audit rows with before/after diffs, document numbering, GST split | `app/api/v1`, `app/services`, `app/models`, `alembic/` |
| 3 Agentic AI | Seven pipeline agents run in a fixed order and pass results along; the learning agent and the assistant sit beside them; every run is logged in `agent_runs` | `app/agents`, `app/ml`, `app/assistant`, `app/llm` |
| 4 Automation | Explicit state machines; every transition validates the move (409 otherwise), writes status history + audit + notifications; approval routing by rules; one-click tokens; Celery Beat schedules | `app/workflow`, `app/tasks` |
| 5 Output | Role dashboards, alerts center, AI insights, reports (PDF/Excel/email), PDFs for POs, GRNs and debit notes, notifications | `frontend/src`, `app/services/{dashboard,reports,documents,notify}.py` |

## Agents and the daily pipeline

```mermaid
flowchart LR
  SM[1 Stock Monitor<br/>positions, low/out/dead stock, ABC/XYZ] --> EX[2 Expiry<br/>sell-through, block expired, markdown/return/transfer]
  EX --> DE[3 Decision<br/>HGB vs MA-28 forecast → SS, ROP, EOQ, qty × correction]
  DE --> SU[4 Supplier<br/>score = 0.35 price + 0.30 on-time + 0.20 lead + 0.15 quality]
  SU --> AO[5 Auto-order<br/>purchase requests → approval engine → grouped POs]
  AO --> TR[6 Tracking<br/>predicted delivery, late risk, reminders, courier]
  TR --> AN[7 Anomaly<br/>5 rules + IsolationForest]
```

Each agent is a class with `run(context) -> AgentResult`; `app/agents/base.execute` runs it in its own transaction,
records `agent_runs`, and catches failures so the next agent still runs. Schedules (Asia/Kolkata): pipeline 06:00,
stock + expiry every 2 h 08:00–22:00, tracking hourly, supplier scores 02:00, learning (accuracy/drift) 05:45,
ABC/XYZ + anomaly model + forecast retraining Sunday 03:00.

## Purchase flow (sequence)

```mermaid
sequenceDiagram
  autonumber
  participant B as Celery Beat
  participant O as Orchestrator
  participant AE as Approval engine
  participant M as Manager
  participant WF as PO workflow
  participant S as Supplier (portal)
  participant ST as Staff
  participant DB as PostgreSQL
  B->>O: 06:00 run pipeline
  O->>DB: forecasts, ROP/SS/EOQ, supplier scores
  O->>AE: create purchase request (reason_json + reason_text)
  AE->>AE: route by rules (≤ ₹5k auto · ≤ ₹50k manager · above: manager → admin)
  AE-->>M: in-app + email (+ Telegram) with single-use Approve / Reject links
  M->>AE: approve (optionally edit qty / supplier)
  AE->>DB: approval + feedback row (learning)
  AE->>WF: create PO (grouped per supplier & location), CREATED → SENT
  WF-->>S: PO PDF (GST split, amount in words) + portal link
  S->>WF: confirm (ETA) · dispatch (tracking ref)
  B->>O: hourly tracking: predicted date, reminders > 24 h, escalation > 48 h
  ST->>WF: GRN (partial allowed, batches + expiry, damaged → quarantine + return draft)
  WF->>DB: batches + purchase_in movements · PO RECEIVED
  S->>WF: invoice (portal / manual / OCR)
  WF->>WF: 3-way match (qty 0 %, price 2 %)
  alt matched
    WF->>DB: INVOICED → CLOSED
  else mismatch
    WF-->>M: critical alert, closing blocked until resolved
  end
```

## State machines (`app/workflow/machines.py`)

| Entity | States |
|---|---|
| Purchase request | DRAFT → PENDING_APPROVAL → APPROVED / REJECTED → ORDERED (+ CANCELLED) |
| Purchase order | CREATED → SENT → CONFIRMED → DISPATCHED → PARTIALLY_RECEIVED → RECEIVED → INVOICED → CLOSED (+ REJECTED_BY_SUPPLIER, CANCELLED) |
| Stock transfer | REQUESTED → APPROVED → DISPATCHED → RECEIVED (+ CANCELLED) |
| Supplier return | DRAFT → PENDING_APPROVAL → APPROVED → DISPATCHED → CREDIT_RECEIVED (+ REJECTED, CANCELLED) |
| RFQ | OPEN → CLOSED → AWARDED (+ CANCELLED) |

## Continuous learning loop

1. Every final approval decision writes `feedback` (approved / edited / rejected).
2. Per product, the correction factor is an EMA (α = 0.3) of final ÷ suggested quantity, clamped to [0.5, 1.5], and
   multiplies the next order quantity.
3. Three rejections in a row lower the product's priority (auto-order then waits until it is actually out of stock)
   and raise an info alert.
4. Daily, the learning agent stores realised MAPE of yesterday's forecast; a rolling 14-day MAPE above 1.5 × the
   training MAPE raises a `forecast_drift` alert.
5. Weekly, a challenger model is trained and compared with the incumbent on the same 28-day holdout; it is promoted
   only if its MAPE is lower. Every version is in `model_registry`.

## Deployment

```mermaid
flowchart LR
  user((Browser / phone)) --> web[web · nginx<br/>PWA build :5173]
  web -- /api --> api[api · FastAPI :8000]
  api --> db[(db · Postgres 16)]
  api --> redis[(redis)]
  worker[worker · Celery] --> db & redis
  beat[beat · Celery Beat] --> redis
  api & worker --> mailpit[mailpit · SMTP :1025 / UI :8025]
```

`docker compose up --build` starts all seven services with health checks; the api container runs `alembic upgrade head`
before serving. Secrets come only from environment variables (`.env.example` documents every one). LLM, Telegram,
IMAP and courier integrations are optional and each has a working fallback.

## Security

JWT access (30 min) + refresh (7 days) tokens revoked through a per-user `token_version`; bcrypt; optional TOTP for
admins; login and forgot-password rate limits; permission checked on every route from one matrix
(`app/core/permissions.py`); suppliers see only their own POs/RFQs/invoices; CORS limited to the frontend origin;
security headers; errors never include stack traces; every create/update/delete/transition is in the audit log with
before/after values and client IP; one-click approval tokens are hashed, single-use and expire after 48 h.
