# TypeSafe Jev / System One — HTTP Wire Schema

Sources: `docs.typesafe.ai` (llms.txt, api.md, primitives/*.md, confidence.md, sdk/python.md)
and the real `typesafe-sdk` Python package (v0.7.0, installed via `uv pip install typesafe-sdk`,
generated from `https://api.typesafe.ai/openapi.json` — this is **documented+code-verified**
ground truth, not inference) plus `system-one-adapter-python` (adapter repo, gives the
confidence-formula reference implementation). Anything not backed by one of these is marked
**[inferred]**.

## 1. Endpoint(s)

| | |
|---|---|
| Method/Path | `POST /v1/systemone` |
| Base URL | `https://api.typesafe.ai` (code: `DEFAULT_BASE_URL`, overridable via `TYPESAFE_BASE_URL`) |
| Auth header | `Authorization: Bearer <API_KEY>` (env `TYPESAFE_API_KEY`) |
| Other req. headers | `Content-Type: application/json`, `Accept: application/json`, `User-Agent: typesafe-sdk/<ver>`, `X-TypeSafe-SDK: typesafe-sdk/<ver>`, `X-TypeSafe-Runtime: python/<ver> (<platform>; <arch>)`, `X-TypeSafe-Retry-Count: <n>` (present only on retries) |
| Resp. header | `x-typesafe-request-id` (also used for `Retry-After`: `retry-after` seconds or `retry-after-ms` on 429) |
| Companion endpoint | `GET /v1/models` → `{"models": [{"name", "description", "release_date"}]}` |
| `model` field values | `"jev-latest"` is the confirmed default/alias (code: `DEFAULT_MODEL = "jev-latest"`, env override `TYPESAFE_DEFAULT_MODEL`). A pinned version string like `"jev-1.13.0"` follows the same `model` field but **[inferred]** — no such literal appears in the docs pages fetched or in the SDK source; only `GET /v1/models` reveals concrete names/aliases at runtime. |

## 2. Request JSON

Top-level (`SystemOneRequest`, all required):

| Field | Type | Notes |
|---|---|---|
| `state` | `string \| object \| array` | Content all questions refer to. |
| `model` | `string` | e.g. `"jev-latest"`. |
| `questions` | `object` (map `string` → Question) | `min_length: 1`. Keys are caller-chosen IDs; answers come back under the same keys. |

Every **Question** is a discriminated union on `type` (`"noul" \| "choice" \| "score"`), extra fields forbidden.

**Noul** (`NoulQuestion`):
- `type`: `"noul"` (required)
- `instructions`: `string | object | array | null` (optional, default `null`) — the yes/no question or statement
- `criteria`: `object | null` (optional, default `null`) — `NoulCriteria`:
  - `true`: `string | object | array | null` (optional) — what counts as yes
  - `false`: `string | object | array | null` (optional) — what counts as no

**Choice** (`ChoiceQuestion`):
- `type`: `"choice"` (required)
- `instructions`: `string | object | array | null` (optional)
- `criteria`: `object` (required) — map of `<option name>` → `string | object | array | null` description. Docs: up to **255 options**; a description-less option is `null` and "interpreted by its name alone." The Python adapter additionally rejects Choice/Score with fewer than 2 criteria client-side; the generated OpenAPI model itself has no `min_length` on Choice criteria.

**Score** (`ScoreQuestion`):
- `type`: `"score"` (required)
- `instructions`: `string | object | array | null` (optional)
- `criteria`: `array` of `string | object | array` (required), ordered low→high, position = score starting at 0. Wire schema constraint from the generated OpenAPI model: `min_length: 1`. Docs (`primitives/score.md`) describe **2–10 ordered levels** as the practical/recommended range — treat 2–10 as documented guidance, `min_length: 1` as the literal wire-level floor **[documented, values differ by source]**.

Structured `instructions`/criteria values (docs `primitives/advanced.md`, **not enforced by the wire schema** — it's just `string | object | array | null`, i.e. free-form JSON):
- Structured instructions commonly use `{name, type, description, unit?}` plus free-form `inspect`, `focus`, `note`, `compare` — **[documented, shape not schema-enforced]**.
- Structured Choice option value: `{"what": ..., "not_for": ..., "examples": [...]}` — **[documented]**.
- Structured Score level entry: `{"summary": ..., "signals": [...]}` — **[documented]**.
- Structured Noul `criteria.true` / `criteria.false`: `{"what": ..., "examples": [...]}` — **[documented]**.

No `state` character/token cap or global question-count cap is documented or present in the generated schema **[gap — not found in any fetched source]**.

### Example 1 — Choice only

```json
{
  "state": "My package arrived damaged and I want a refund.",
  "model": "jev-latest",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this ticket?",
      "criteria": {
        "returns": "Damaged, wrong, or unwanted item returns and refunds",
        "shipping": "Delivery delays, lost packages, address changes",
        "billing": "Charges, invoices, payment methods"
      }
    }
  }
}
```

### Example 2 — Score only

```json
{
  "state": "The server has been down for 2 hours and customers cannot check out.",
  "model": "jev-latest",
  "questions": {
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this issue?",
      "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]
    }
  }
}
```

### Example 3 — Noul only

```json
{
  "state": "Hi, I'd like to cancel my subscription and get my money back.",
  "model": "jev-latest",
  "questions": {
    "refund_requested": {
      "type": "noul",
      "instructions": "The customer is requesting a refund.",
      "criteria": {
        "true": "Explicitly asks for money back or a refund",
        "false": "No mention of a refund"
      }
    }
  }
}
```

### Example 4 — Mixed batch (all three types, one request, one `state`)

```json
{
  "state": {
    "subject": "Duplicate charge",
    "message": "I was charged twice for order #4471. Please fix this ASAP, this is ridiculous."
  },
  "model": "jev-latest",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this ticket?",
      "criteria": {
        "returns": "Damaged, wrong, or unwanted item returns and refunds",
        "billing": "Charges, invoices, payment methods"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this message?",
      "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]
    },
    "is_angry": {
      "type": "noul",
      "instructions": "The customer sounds angry or frustrated."
    }
  }
}
```

## 3. Response JSON

Top-level (`SystemOneResponse`), all required:

| Field | Type | Notes |
|---|---|---|
| `model` | `string` | Model that actually answered; may differ from a requested alias. |
| `answers` | `object` (map `string` → Answer) | `min_length: 1`, keyed by the question IDs from the request. |
| `usage` | `object` | `Usage`: `{"input_tokens": int, "output_tokens": int}`. Docs note output tokens are "currently free of charge." |

No response-body `id` or timing field is defined by the wire schema; the only per-request identifier is the `x-typesafe-request-id` **response header**. (Latency/retry counters such as `n_retries`, `latency`, `input_tokens_total` are adapter-only additions in `system-one-adapter-python`'s `Usage`/`debug` fields — **not part of the real Jev API response**.)

**Answer** is a discriminated union on `type`:

**Noul** (`NoulAnswer`):
- `type`: `"noul"`
- `noul`: `float` [0,1] — probability the answer is yes/true

**Choice** (`ChoiceAnswer`):
- `type`: `"choice"`
- `choice`: `string` — name of the highest-probability option
- `confidence`: `float` [0,1]
- `probabilities`: `object` (map option name → `float`), sums to ≈1

**Score** (`ScoreAnswer`):
- `type`: `"score"`
- `score`: `float` — probability-weighted average level, may be fractional
- `confidence`: `float` [0,1]
- `legend`: `object` (map level-index-as-string → description, echoing the request's `criteria`)
- `probabilities`: `object` (map level-index-as-string → `float`), sums to ≈1

### Example response (for mixed batch above)

```json
{
  "model": "jev-latest",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.82,
      "probabilities": { "returns": 0.09, "billing": 0.91 }
    },
    "urgency": {
      "type": "score",
      "score": 1.7,
      "confidence": 0.65,
      "legend": { "0": "Can wait", "1": "Needs attention this week", "2": "Needs attention today" },
      "probabilities": { "0": 0.05, "1": 0.25, "2": 0.7 }
    },
    "is_angry": {
      "type": "noul",
      "noul": 0.87
    }
  },
  "usage": { "input_tokens": 142, "output_tokens": 31 }
}
```

## 4. Confidence formulas

Confidence is defined for **Choice** and **Score**; **Noul has no confidence field at all** (docs, and `NoulAnswer` has no `confidence` in the wire schema). Formulas below are the exact reference implementation in `system_one_adapter/_utils/confidence_metrics.py` (the adapter this task targets); TypeSafe's own docs state only that confidence is "derived from probability distribution shape" without publishing a formula, so these are the concrete, code-verified version to replicate:

**Choice** — scale the peak probability from uniform (0) to certain (1):
```
K = number of options
c = (max(probabilities) − 1/K) / (1 − 1/K)
```
(`K == 1` ⇒ `c = 1.0`; probabilities are re-normalized to sum to 1 first, falling back to uniform if they sum to 0.)

**Score** — 1 minus the probability-weighted mean absolute distance from the modal level, normalized against a uniform distribution's own mean absolute deviation:
```
mode = argmax(probabilities)
distance = Σ_i probabilities[i] * |i − mode|
uniform_center = (n − 1) / 2
uniform_mad = (Σ_i |i − uniform_center|) / n      # n = number of levels
c = max(0, 1 − distance / uniform_mad)
```
(`n == 1` ⇒ `c = 1.0`; probabilities re-normalized the same way as Choice.)

**Score value** itself (not confidence): expected value over the (rescaled-to-sum-1) level probabilities —
```
score = Σ_i i * probabilities[i]
```
matching docs' plain-language description ("each level number multiplied by its probability, added up").

**Noul**: no confidence; the single `noul` value already is the probability of "yes."

## 5. Error format

HTTP status → SDK exception mapping (`typesafe_sdk._core.errors`):

| Status | Meaning |
|---|---|
| 400 | Bad request |
| 401 | Authentication failed (missing/invalid API key) |
| 403 | Permission denied |
| 404 | Not found |
| 422 | Unprocessable Entity — request validation failed |
| 429 | Rate limit exceeded (`retry-after` / `retry-after-ms` headers carry backoff) |
| 5xx | Internal server error |

**422 body** is FastAPI/Pydantic-style and *is* part of the generated OpenAPI schema (`HTTPValidationError`):
```json
{
  "detail": [
    {
      "loc": ["body", "questions", "urgency", "score", "criteria"],
      "msg": "Field required",
      "type": "missing",
      "input": { "type": "score" },
      "ctx": { "min_length": 1 }
    }
  ]
}
```
`ValidationError` fields: `loc` (`array<string|int>`), `msg` (`string`), `type` (`string`, machine-readable code e.g. `"missing"`), `input` (`any`, optional), `ctx` (`object`, optional).

Other error statuses do **not** have a schema in the OpenAPI spec; the SDK defensively looks for, in order, a top-level `error` (string or `{"message": str}`), `message` (string), `detail` (string, `{"message": str}`, or the FastAPI list form above) — **[inferred from SDK's tolerant parsing, exact non-422 error body shape not documented]**. Falls back to raw response text truncated to 200 chars, or `"status code (no body)"` if the body is empty. All API errors expose `status`, `body`, `headers`, and the `x-typesafe-request-id` header via `.request_id`.
