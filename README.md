# free-model-aggregator-01

One Python 3.13 standard-library-only server (`main.py`) that serves the
OpenAI Chat Completions API (`POST /v1/chat/completions`) and the Anthropic
Messages API (`POST /v1/messages`), in JSON and in server-sent events,
backed by the simulated free-tier providers of a make.fun trial.

## How it works

- Reads the trial brief from `$MAKEFUN_PROMPT_FILE` (provider list, quality
  table, workload) and the trial environment from `$MAKEFUN_ENV_URL`.
- Probes every provider's catalog, registers the port and the catalog with
  the environment, then serves the 20 trial requests.
- Routing is risk-aware: each provider's k-th call carries a prior refusal
  probability (call #1 is always safe; later calls can hit the provider's
  scheduled refusals, its outage, or its spent budget — exact per-trial
  numbers are never known and never used, only the published prior). Each
  pick maximises entry weight times rating minus 400 milli per unit of
  refusal risk, so a slightly weaker model on a likely-safe next call beats
  the top model on a likely-refusing one: every refusal costs an upstream
  call, and the final ranking breaks exact score ties by call count. The
  risk is the exact Bayesian posterior over the published prior,
  conditioned on that provider's own observed outcomes so far in the
  trial: a refusal raises the next-call risk a lot, and a success on a
  small-budget provider raises it too (its scheduled failures must sit at
  later call numbers). No separate suspend penalty is needed — the
  posterior already encodes the failure information. After three
  consecutive refusals a provider is dropped for the rest of the trial
  (past its outage or budget). On 402 / 429 / 5xx it fails over to the
  next candidate without changing the request's semantics (system prompt,
  tools, JSON mode, images, max_tokens). Quality is guarded: once the
  weighted shortfall served so far exceeds 550 milli, the router switches
  to rating-first for the rest of the trial, so the score-99 cliff
  (shortfall budget 764 milli) is not breached. Score 99 needs quality
  >= 98, so the router accepts quality 98 to buy fewer calls.
- Deterministic: fixed timestamps, no wall clock in any decision, no
  unseeded randomness.
- `python3 main.py --list` prints the catalog document byte-for-byte as
  registered; outside a trial (no `$MAKEFUN_PROMPT_FILE`, no
  `$MAKEFUN_ENV_URL`) it prints `{"providers":[]}` and exits 0.

## Layout

- `main.py` — the whole server, standard library only.
- `LICENSE` — MIT.
- `.makefun/owner.json` — wallet claim for the make.fun account.

## Local verification (kit practice trials)

Practice trials 0, 1 and 2, scored with the kit's own scorer:

| trial | score | quality | served | wire | failover | catalog | upstream calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0 | 99 | 98 | 100 | 100 | 100 | 100 | 23 |
| 1 | 99 | 98 | 100 | 100 | 100 | 100 | 22 |
| 2 | 99 | 98 | 100 | 100 | 100 | 100 | 22 |

The previous rating-first router scored 99 / 99 / 99 with 26 / 30 / 30
calls (mean 28.7). Iteration 2's risk-aware router (rating - 250 * prior
risk, suspend penalty 300, drop after 2) scored 99 / 99 / 99 with 23 / 22
/ 24 calls (mean 23.0). Iteration 3's router — weight * rating - 400 *
Bayesian posterior risk, no suspend penalty, drop after 3, quality guard
550 — scores 99 / 99 / 99 with 23 / 22 / 22 calls (mean 22.33). The
target of <= 21 calls mean was NOT reached: on 10,000 fresh random
instances the same policy averages 22.67 calls, P(all 3 trials score 99)
= 0.983, P(score 99 and <= 21 calls on one trial) = 0.281, P(mean of 3
trials <= 22) = 0.342, P(mean <= 21) = 0.029. The barrier is structural:
every non-first call to a provider carries a ~25-45% refusal risk and 10
such calls are needed; an oracle that knows each instance's refusals in
advance still averages 21.71 calls, and only by sacrificing quality
(P(score 99) = 0.601). The 20-call entries on the public dev board mostly
sit at quality 98 too, but that board runs on the public practice
instance — the final runs on 3 fresh scored instances, where only the
policy's distribution matters. Re-running trial 0 produces a byte-identical
state export (deterministic replay verified, sha256
bbc4ab27b01ada006ef5db7dc47dfc7e833a9ceb0dae688a0070b1184526d430).

Status: FIXING (iterasi 3 selesai) — LOCALLY VERIFIED, OFFICIAL RESULT NOT
YET VERIFIED. The house scored run happens server-side at submission; no
official score is claimed.
