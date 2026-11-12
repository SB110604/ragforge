# SmartStock RPA — User guide

Demo store: **Shree Ganesh SuperMart, Kothrud, Pune** (created by `make seed`). Open http://localhost:5173.

| Role | Login | Password |
|---|---|---|
| Admin (owner) | admin@demo.in | Admin@123 |
| Manager | manager@demo.in | Manager@123 |
| Staff | staff@demo.in | Staff@123 |
| Supplier (Agarwal Wholesale Traders) | supplier1@demo.in | Supplier@123 |
| Supplier (Shah Distributors) | supplier2@demo.in | Supplier@123 |

Everywhere: the top bar has search (products, SKU, barcode or a PO number), the notification bell, the assistant
(chat bubble), language (English / हिन्दी / मराठी) and dark mode. On a phone, the menu is behind ☰; the app can be
installed from the browser menu ("Install app" / "Add to Home screen").

---

## Staff

**Dashboard** — today's work: POs to receive, stock counts in progress, expiring items to move, big buttons for
*Scan & sell*, *Scan & receive* and *Stock counts*.

**Sell (Sales → Billing)**
1. Click the search box (or press *Scan*) and scan the product barcode with the laptop/phone camera; a USB barcode
   scanner also works — it types the code and presses Enter.
2. Adjust quantity or discount; scanning the same item again adds one.
3. *Complete sale* — the earliest-expiry batch is used automatically (FEFO) and markdown prices apply. Print the receipt.

**Receive stock without a PO (Stock → Stock in)** — scan, enter quantity, batch and expiry (pre-filled for
perishables from the shelf life), *Receive stock*.

**Receive a delivery (Purchase orders → open the PO → Receive goods (GRN))**
1. Quantities default to what is still due; change them for a partial delivery.
2. Enter the batch number and expiry (scan a batch label to fill it). Damaged units go in *Damaged* — they are kept
   in a blocked quarantine batch and a supplier-return draft is created.
3. *Post goods receipt* — stock increases immediately; the PO becomes *Received* (or *Partially received*).

**Stock count (Stock → Stock counts)** — *Start count* (optionally one category), enter or scan counted quantities,
*Save*. A manager completes it, which posts the variances as adjustments. **Transfers** — request stock from the
warehouse; after a manager approves, *Dispatch* at the source and *Receive* at the destination.

**Expiry** — see what must be moved first; staff can view, managers act.

## Manager

**Dashboard** — KPIs (stock value, below reorder point, out of stock, near-expiry value, pending approvals, open and
late POs, purchases this month, turnover, AI acceptance), charts (sales vs forecast, stock value trend, top movers,
expiry risk by week, supplier on-time %), latest alerts and approvals waiting for you.

**Approve purchases (Purchase requests)**
1. *Waiting for me* lists requests at your approval level (🤖 = AI suggestion).
2. Open one: the *Why the AI suggests this* panel shows the forecast, lead time, safety stock, reorder point, EOQ,
   pack rounding, learned correction and the supplier score breakdown.
3. Approve as is, change the quantity or supplier and approve, or reject with a reason. Tick several and *Approve
   selected* for bulk approval. Approved requests become a PO that is emailed to the supplier automatically.
4. You can also approve from the email (or Telegram) link — one click to approve, a reason to reject.

**Purchase orders** — status chips, late flag, timeline, PDF, *Resend*, *Cancel*, GRNs, invoices. *Enter invoice*
can be filled by **scanning the supplier's bill** (OCR) — check the per-line confidence, then save. The 3-way match
runs automatically; a mismatch blocks closing until you resolve it with a note.

**Expiry** — expired / critical / warning batches with the agent's suggestion and buttons: *Markdown* (applies at
the billing counter), *Transfer*, *Return* (creates a supplier return), *Write off*.

**Returns** — submit → approve (debit-note PDF emailed) → mark dispatched (stock leaves) → record credit note.

**RFQs** — *New RFQ*: products, quantities, suppliers → emailed with a portal link. Compare the quotes (lowest is
tagged, saving vs current price) and *Use this quote* to update the supplier's price list.

**Suppliers** — score breakdown chart, on-time trend, price list (add/edit), PO history.

**AI Insights** — agent cards with last results and *Run now* / *Run full pipeline*, run history, 30-day forecast
per product with interval, ABC/XYZ matrix, dead stock, anomalies (*Mark valid* / *Investigate*), AI performance
(MAPE trend, acceptance %, average correction, model versions) and the model registry.

**What-if simulator** — change demand %, lead time, make a supplier unavailable or change the service level to see
new SS / ROP / EOQ / order quantities, stock-out risk and cost. Nothing is saved.

**Reports** — stock valuation, sales, purchases, supplier performance, expiry & wastage, GST summary; date filter;
PDF or Excel; *Schedule a report* to email it daily / weekly / monthly.

**Import center** — download a template, upload CSV/XLSX, check the column mapping, *Validate & preview* (row
errors shown), import the valid rows, download the error report.

**Labels** — select products or batches, number of copies, print an A4 sheet of Code128 + QR labels.

## Admin

Everything a manager can do, plus: **Settings** — business profile and integrations status, parameters (ordering
cost, holding cost, service level, review period, expiry thresholds, markdowns, supplier score weights, 3-way
tolerances, notification channels, PO terms), users (roles, supplier links, passwords), locations & categories,
approval rules (with a "try an amount" preview), POS API keys, backups (back up now, download, restore with typed
confirmation). **Audit log** — every change with before/after diff. **System status** — DB, Redis, worker, beat.
**Profile** — set up two-factor authentication. Approvals above ₹50,000 need the admin after the manager.

## Supplier

**Portal** — open orders, orders awaiting confirmation, late orders, RFQs awaiting a quote and your score.
Open a PO: download the PDF, *Confirm order* (with ETA) or *Reject order* (reason), *Dispatch details* (ETA,
courier, tracking/LR number), *Upload invoice* (PDF/photo). **Requests for quotation** — enter price and lead time
per item and submit. **My invoices** — status of each invoice (matched / mismatch). You only ever see your own data.

## Assistant

Ask in plain language, e.g. "How much rice is left?", "Which items expire this week?", "Show pending approvals",
"Best supplier for sugar?", "What is the forecast for milk?", "Sales this week", "Create a purchase request for 50 kg
sugar" (creates a DRAFT — submit it on the Purchase requests page). Without an LLM configured, a rule-based matcher
answers using the same read-only tools; with `LLM_PROVIDER` set it uses tool calling and falls back automatically.

---

## 10-minute viva demo script

Before the viva: `make up`, `make seed` (≈ 1–2 min), open http://localhost:5173 and http://localhost:8025 (Mailpit)
in two tabs. Keep `reports/evaluation.md` open.

| Time | Do | Say |
|---|---|---|
| 0:00 | Log in as **manager**. Show the dashboard. | "Software-only intelligent RPA for a supermarket: five layers — data input, processing, agentic AI, automation, output." Point at KPIs and *Sales vs forecast*. |
| 1:00 | AI Insights → *Run full pipeline*. Show the result chips. | "Seven agents run in order every morning at 06:00 and hand results to each other: stock, expiry, decision, supplier, auto-order, tracking, anomaly." |
| 2:00 | Purchase requests → open an AI request. | Walk through *Why the AI suggests this*: forecast, SS = z·σ·√L, ROP, EOQ, pack rounding, learned correction, supplier score bars. |
| 3:00 | Change the supplier to **Agarwal Wholesale Traders**, *Approve*. Open the PO. | "Approval rules route by amount; approval auto-creates the PO with CGST/SGST or IGST and emails it." Show the PDF and the email in Mailpit (one-click links in approval emails too). |
| 4:30 | Log in as **supplier1** → portal → open the PO → *Confirm*, then *Dispatch details*. | "Suppliers have their own login and see only their orders; the timeline records every step." |
| 5:30 | Log in as **staff** → open the PO → *Receive goods (GRN)* → post. Open the product page. | "Batches with expiry are created, stock increases, FEFO applies at billing." |
| 6:30 | **Expiry** page → apply a suggested markdown; **Alerts** → anomalies. | "Expired stock is blocked; markdown/transfer/return suggestions; anomalies combine rules with an IsolationForest." |
| 7:30 | Open the chat bubble: "Which items expire this week?", "Create a purchase request for 50 kg sugar". | "The assistant only calls read-only tools and can draft — never approve — a request; it works with or without an LLM." |
| 8:15 | AI Insights → *AI performance*; What-if simulator (+30 % demand). | "Approver edits feed a correction factor; weekly retraining promotes a model only if MAPE improves; drift raises an alert." |
| 9:00 | Show `reports/evaluation.md`. | Quote forecast MAPE vs moving average and the policy table (stock-outs, fill rate, waste, inventory, time saved) — and the inventory trade-off. |
| 9:40 | Switch language to हिन्दी / मराठी, toggle dark mode, show it on a phone. | "Role-based, multilingual, installable PWA; everything is audited." |
