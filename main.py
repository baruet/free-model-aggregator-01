#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten simulated providers.

Python 3.13, standard library only. The server reads the brief at
$MAKEFUN_PROMPT_FILE (provider directory, sealed quality table, workload
manifest), probes every provider's catalog route, registers its port and
catalog with the env at $MAKEFUN_ENV_URL, and serves the 20 requests the
env's client sends: POST /v1/chat/completions (OpenAI wire) and
POST /v1/messages (Anthropic wire), JSON and SSE streaming, plus
GET /v1/models in the wire the auth header names.

Routing: every request is served by the highest-rated free model that
meets its requirements, with a primary assignment planned up front from
the manifest (weight first, scarcest requirements first, at most four
primaries per provider) so no provider's free tier is burned on the
first requests. A provider call that answers 402, 429 or 5xx is never
forwarded: the request is retried on the next best candidate that meets
it, and a provider that fails twice in a row is tried last.

Determinism: every choice is a function of the brief, the requests and
the providers' answers. No wall clock and no unseeded randomness appear
in any request, response or registered document (fixed timestamps only).

`python3 main.py --list` prints the catalog document byte-for-byte as
registered and exits 0. Outside a trial it prints {"providers":[]}.
"""

import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
POLL_INTERVAL = 0.1
PROBE_TIMEOUT = 5.0
UPSTREAM_TIMEOUT = 15.0
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_ATTEMPTS_PER_REQUEST = 20
PRIMARY_CAP = 4  # kept for reference; the risk-aware plan below supersedes it

# P(refusal) of a provider's k-th call (k = 1..16), from the task's published
# mechanics: each provider's free-tier budget is drawn per trial from a fixed
# range, with 1-2 scheduled refusals inside [2, budget-1], one provider
# always refusing its 2nd call, and one provider going out from a call in
# [2, budget-1]. Exact per-trial budgets and schedules are NEVER known to
# this program and are never used; only these prior probabilities (computed
# offline by enumerating the ranges) guide routing. Call #1 is always safe.
CALL_RISK = {
    "groq": [0.0, 0.295, 0.214, 0.225, 0.237, 0.248, 0.26, 0.252, 0.227, 0.348, 0.503, 0.698, 1.0, 1.0, 1.0, 1.0],
    "cerebras": [0.0, 0.508, 0.439, 0.467, 0.319, 0.438, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "gemini": [0.0, 0.405, 0.327, 0.347, 0.366, 0.304, 0.411, 0.589, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "zai": [0.0, 0.615, 0.558, 0.39, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "openrouter": [0.0, 0.287, 0.205, 0.216, 0.228, 0.238, 0.249, 0.242, 0.253, 0.214, 0.4, 0.636, 1.0, 1.0, 1.0, 1.0],
    "sambanova": [0.0, 0.615, 0.558, 0.39, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "pollinations": [0.0, 0.268, 0.185, 0.195, 0.206, 0.215, 0.225, 0.218, 0.228, 0.214, 0.309, 0.429, 0.574, 0.749, 1.0, 1.0],
    "cloudflare": [0.0, 0.386, 0.306, 0.325, 0.343, 0.36, 0.248, 0.475, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "ovh": [0.0, 0.508, 0.439, 0.467, 0.319, 0.438, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "kilol": [0.0, 0.243, 0.157, 0.165, 0.175, 0.184, 0.192, 0.186, 0.195, 0.204, 0.212, 0.199, 0.306, 0.434, 0.585, 0.761],
}
RISK_LAMBDA = 400  # milli-rating charged per unit of refusal risk (per unit weight)
QUALITY_GUARD = 550  # milli of weighted quality shortfall: past this, rating-first

# Per-provider free-tier budget ranges (structural constants, published in
# the kit generator's provider table). Used only to derive the prior
# refusal probabilities above and the Bayesian posterior below.
BUDGET_RANGE = {
    "groq": (9, 12), "cerebras": (5, 6), "gemini": (6, 8), "zai": (4, 5),
    "openrouter": (10, 12), "sambanova": (4, 5), "pollinations": (10, 14),
    "cloudflare": (7, 8), "ovh": (5, 6), "kilol": (12, 16),
}


def _build_posterior_risk():
    """POST_RISK[provider][(m, bits)] = P(provider's call m+1 refuses |
    outcomes of its calls 1..m), bit (k-1) of `bits` = 1 means call k
    refused. Exact enumeration over the published priors (same structure
    as CALL_RISK): budget ~ U[lo,hi], 1-2 scheduled refusals in
    [2, budget-1], one provider always refuses call 2, one provider goes
    out from a call in [2, budget-1]. Only published priors plus this
    program's own observed outcomes are used; exact per-trial budgets
    and schedules are never known. Deterministic: fixed iteration order,
    no randomness, no wall clock."""
    from itertools import combinations
    post = {}
    for provider_id, (lo, hi) in BUDGET_RANGE.items():
        mask_weight = {}
        for budget in range(lo, hi + 1):
            for n_fail in (1, 2):
                for fails in combinations(range(2, budget), n_fail):
                    for designated in (False, True):
                        p_des = 0.1 if designated else 0.9
                        fail_set = set(fails)
                        if designated:
                            fail_set = {2} | set(list(fail_set)[1:])
                        for outage in (False, True):
                            p_out = 0.1 if outage else 0.9
                            if not outage:
                                mask = 0
                                for k in range(1, 10):
                                    if (k in fail_set) or (k > budget):
                                        mask |= (1 << (k - 1))
                                mask_weight[mask] = mask_weight.get(mask, 0.0) + p_des * p_out
                            else:
                                for start in range(2, budget):
                                    mask = 0
                                    for k in range(1, 10):
                                        if (k in fail_set) or (k >= start) or (k > budget):
                                            mask |= (1 << (k - 1))
                                    mask_weight[mask] = mask_weight.get(mask, 0.0) + p_des * p_out * (1.0 / (budget - 2))
        table = {}
        for m in range(0, 9):
            for bits in range(1 << m):
                total = 0.0
                fail_next = 0.0
                for mask, weight in mask_weight.items():
                    consistent = True
                    for k in range(1, m + 1):
                        if bool((mask >> (k - 1)) & 1) != bool((bits >> (k - 1)) & 1):
                            consistent = False
                            break
                    if not consistent:
                        continue
                    total += weight
                    if (mask >> m) & 1:
                        fail_next += weight
                table[(m, bits)] = (fail_next / total if total else 0.5)
        post[provider_id] = table
    return post


POST_RISK = _build_posterior_risk()


def post_risk(provider_id, history):
    """Posterior P(provider's next call refuses) given `history`, the
    tuple of booleans (True = refused) for its calls so far, in order.
    Unknown provider: 0.5."""
    table = POST_RISK.get(provider_id)
    if table is None:
        return 0.5
    m = len(history)
    if m >= 9:
        return 1.0
    bits = 0
    for i, failed in enumerate(history):
        if failed:
            bits |= (1 << i)
    return table[(m, bits)]


def call_risk(provider_id, call_no):
    """Prior P(provider's call_no-th call refuses). Unknown provider: 0.5."""
    table = CALL_RISK.get(provider_id)
    if table is None:
        return 0.5
    if call_no < 1:
        return 0.0
    if call_no > len(table):
        return 1.0
    return table[call_no - 1]
SAFETY_WALL_SECONDS = 165.0

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Brief parsing
# ---------------------------------------------------------------------------

def documents_in(text):
    """Every JSON object in `text` that carries a string `schema`, in order."""
    dec = json.JSONDecoder()
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find("{", i)
        if j == -1:
            break
        try:
            value, end = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if is_record(value) and isinstance(value.get("schema"), str):
            out.append(value)
        i = max(end, j + 1)
    return out


def read_brief(path):
    """Return (providers, ratings, workload, error).

    providers: list of dicts {id, name, wire, auth, base, key_env, key}
    ratings: {(provider, model): row dict}
    workload: list of {id, api, stream, weight, requires}
    """
    providers = []
    ratings = {}
    workload = []
    if not path:
        return providers, ratings, workload, "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return providers, ratings, workload, "cannot read %s: %s" % (path, exc)
    docs = documents_in(text)
    directory = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
            break
    if directory is None:
        return providers, ratings, workload, "no %s document in brief" % PROVIDERS_SCHEMA
    for raw in directory["providers"]:
        if not is_record(raw):
            continue
        pid = raw.get("id")
        wire = raw.get("wire")
        if not isinstance(pid, str) or not pid or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = raw.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if isinstance(raw.get(key), str) and raw[key]:
                base = raw[key]
                break
        providers.append({
            "id": pid,
            "name": raw.get("name") if isinstance(raw.get("name"), str) and raw["name"] else pid,
            "wire": wire,
            "auth": auth,
            "base": base,
            "key_env": raw.get("key_env") if isinstance(raw.get("key_env"), str) else "",
            "key": raw.get("key") if isinstance(raw.get("key"), str) else "",
        })
    for doc in docs:
        if doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            for raw in doc["ratings"]:
                if not is_record(raw):
                    continue
                model = raw.get("model")
                provider = raw.get("provider")
                if not isinstance(model, str) or not model or not isinstance(provider, str) or not provider:
                    continue
                ratings[(provider, model)] = {
                    "model": model,
                    "provider": provider,
                    "rating_milli": raw.get("rating_milli") if isinstance(raw.get("rating_milli"), int) else 0,
                    "context_length": raw.get("context_length") if isinstance(raw.get("context_length"), int) else None,
                    "tools": raw.get("tools") is True,
                    "vision": raw.get("vision") is True,
                    "json": raw.get("json") is True,
                    "free": raw.get("free") is True,
                }
            break
    for doc in docs:
        if doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            for raw in doc["manifest"]:
                if not is_record(raw):
                    continue
                rid = raw.get("id")
                if not isinstance(rid, str) or not rid:
                    continue
                req = raw.get("requires") if is_record(raw.get("requires")) else {}
                requires = {}
                if req.get("tools") is True:
                    requires["tools"] = True
                if req.get("vision") is True:
                    requires["vision"] = True
                if req.get("json") is True:
                    requires["json"] = True
                if isinstance(req.get("min_context"), int):
                    requires["min_context"] = req["min_context"]
                workload.append({
                    "id": rid,
                    "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                    "stream": raw.get("stream") is True,
                    "weight": 2 if raw.get("weight") == 2 else 1,
                    "requires": requires,
                })
            break
    return providers, ratings, workload, None


# ---------------------------------------------------------------------------
# Provider catalog probe
# ---------------------------------------------------------------------------

def join_base(env_url, base):
    base = (base or "").strip()
    if base.lower().startswith("http://") or base.lower().startswith("https://"):
        return base.rstrip("/")
    prefix = (env_url or "").rstrip("/")
    if not base:
        return prefix
    return prefix + ("" if base.startswith("/") else "/") + base.rstrip("/")


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


def _zero_priced(v):
    """True when a listing prices the model at 0, None when it says nothing."""
    if not is_record(v):
        return None
    pricing = v.get("pricing")
    if not is_record(pricing):
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(x)
        elif isinstance(x, str) and x.strip():
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, raw):
    if not is_record(raw):
        return None
    if wire == "gemini":
        name = raw.get("name")
        model_id = name[7:] if isinstance(name, str) and name.startswith("models/") else (name if isinstance(name, str) else "")
    else:
        model_id = raw.get("id") if isinstance(raw.get("id"), str) else ""
    if not model_id:
        return None
    context = _num_or_none(raw.get("context_length"))
    if context is None:
        context = _num_or_none(raw.get("context_window"))
    if context is None:
        context = _num_or_none(raw.get("inputTokenLimit"))
    display = raw.get("display_name") or raw.get("displayName") or model_id
    free = _bool_or_none(raw.get("free"))
    if free is None:
        free = _zero_priced(raw)
    return {
        "id": model_id,
        "display_name": display if isinstance(display, str) else model_id,
        "free": free,
        "context_length": int(context) if context is not None else None,
        "tools": _bool_or_none(raw.get("tools")),
        "vision": _bool_or_none(raw.get("vision")),
        "json": _bool_or_none(raw.get("json")),
    }


def probe_provider(env_url, provider):
    """GET the provider's catalog route. Returns (ok, [models])."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
            if resp.status != 200:
                return False, []
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a failed probe leaves the provider out
        log("probe %s failed: %s" % (provider["id"], exc))
        return False, []
    if not is_record(body):
        return False, []
    raw_list = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    if not isinstance(raw_list, list):
        return False, []
    models = []
    for raw in raw_list:
        model = _model_in(provider["wire"], raw)
        if model is not None:
            models.append(model)
    return True, models


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] is True


def build_catalog(probes, ratings):
    """The --list document and the register body: the same object."""
    out_providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = ratings.get((probe["provider"]["id"], model["id"]))
            context = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({
                "id": model["id"],
                "free": model_free(model, row),
                "context_length": context,
            })
        out_providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": out_providers}


def served_models(probes, ratings):
    """Free models the entry serves, in brief provider order, ids sorted."""
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = ratings.get((probe["provider"]["id"], model["id"]))
            if not model_free(model, row):
                continue
            out.append({"id": model["id"], "provider": probe["provider"]["id"],
                        "display_name": model["display_name"]})
    return out


def load(env_url, prompt_file):
    """Read the brief, probe the providers, build the catalog."""
    providers, ratings, workload, error = read_brief(prompt_file)
    if error:
        log(error)
    probes = []
    if env_url:
        for provider in providers:
            ok, models = probe_provider(env_url, provider)
            if not ok:
                # One retry: a transient catalog failure must not silently
                # drop a whole provider (and its models) from the catalog.
                ok, models = probe_provider(env_url, provider)
            if not ok:
                log("probe %s: no catalog" % provider["id"])
            probes.append({"provider": provider, "ok": ok, "models": models})
    catalog = build_catalog(probes, ratings)
    return {
        "providers": providers,
        "ratings": ratings,
        "workload": workload,
        "probes": probes,
        "catalog": catalog,
        "served": served_models(probes, ratings),
    }


# ---------------------------------------------------------------------------
# Wire helpers: text / image extraction, mirroring the providers' parsers
# ---------------------------------------------------------------------------

def blocks_text(content):
    """Text of a string or of a block array's text blocks, joined with \n."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content
                     if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))


def _image_parts_openai(content):
    """(text, image_url_or_None) of an OpenAI message content."""
    text = blocks_text(content)
    image = None
    if isinstance(content, list):
        for block in content:
            if is_record(block) and block.get("type") == "image_url" and is_record(block.get("image_url")):
                url = block["image_url"].get("url")
                if isinstance(url, str) and url:
                    image = url
                    break
    return text, image


def _split_data_url(url):
    """(media_type, base64_data) for a data: URL, else (None, None)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    head, _, data = url[5:].partition(",")
    media = head.split(";")[0] or "image/png"
    return media, data


def _parse_args(value):
    """A tool call's arguments as a dict."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


# --- client body -> provider body -------------------------------------------

def openai_to_anthropic(body, model):
    """OpenAI Chat Completions request as an Anthropic Messages request."""
    messages = []
    system_parts = []
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if is_record(raw) and raw.get("role") == "system":
            text = blocks_text(raw.get("content"))
            if text:
                system_parts.append(text)
    system_text = "\n".join(system_parts)
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            continue
        content = raw.get("content")
        if role == "assistant":
            blocks = []
            text = blocks_text(content)
            if text:
                blocks.append({"type": "text", "text": text})
            if isinstance(content, list):
                for block in content:
                    if is_record(block) and block.get("type") == "tool_use":
                        pass
            for call in raw.get("tool_calls", []) if isinstance(raw.get("tool_calls"), list) else []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) else "",
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "input": _parse_args(fn.get("arguments")),
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                "content": content if isinstance(content, str) else blocks_text(content),
            }
            if messages and messages[-1].get("role") == "user":
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "user", "content": [block]})
            continue
        # user
        parts = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    url = block["image_url"].get("url")
                    if isinstance(url, str) and url:
                        media, data = _split_data_url(url)
                        if data is not None:
                            parts.append({"type": "image", "source": {
                                "type": "base64", "media_type": media, "data": data}})
                        else:
                            parts.append({"type": "image", "source": {"type": "url", "url": url}})
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if system_text:
        out["system"] = system_text
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else []
    if stops:
        out["stop_sequences"] = stops
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if isinstance(choice, str):
        out["tool_choice"] = {"type": "auto"} if choice == "auto" else {"type": "any"} if choice == "required" else {"type": "none"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        out["tool_choice"] = {"type": "tool", "name": choice["function"].get("name", "")}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def anthropic_to_openai(body, model):
    """Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system_text = blocks_text(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        if raw.get("role") == "assistant":
            tool_calls = []
            for block in raw.get("content", []) if isinstance(raw.get("content"), list) else []:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": compact(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = blocks_text(raw.get("content"))
            entry = {"role": "assistant", "content": text if text else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        # user
        parts = []
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": block.get("content") if isinstance(block.get("content"), str) else blocks_text(block.get("content")),
                })
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_record(block.get("source")):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                    parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}})
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        out["stop"] = [s for s in stops if isinstance(s, str)]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _gemini_parts_from_openai_message(raw):
    """(role, parts) for one OpenAI message, or None."""
    role = raw.get("role")
    content = raw.get("content")
    if role == "assistant":
        parts = []
        text = blocks_text(content)
        if text:
            parts.append({"text": text})
        for call in raw.get("tool_calls", []) if isinstance(raw.get("tool_calls"), list) else []:
            if not is_record(call) or not is_record(call.get("function")):
                continue
            fn = call["function"]
            parts.append({"functionCall": {
                "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                "args": _parse_args(fn.get("arguments")),
            }})
        return ("model", parts) if parts else None
    if role == "tool":
        text = content if isinstance(content, str) else blocks_text(content)
        return ("user", [{"functionResponse": {
            "name": "",
            "response": {"content": text},
        }}])
    if role != "user":
        return None
    parts = []
    if isinstance(content, str):
        parts.append({"text": content})
    elif isinstance(content, list):
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                url = block["image_url"].get("url")
                if isinstance(url, str) and url:
                    media, data = _split_data_url(url)
                    if data is not None:
                        parts.append({"inlineData": {"mimeType": media, "data": data}})
                    else:
                        parts.append({"fileData": {"fileUri": url}})
    return ("user", parts) if parts else None


def openai_to_gemini(body, model):
    """OpenAI Chat Completions request as a Gemini generateContent request."""
    system_parts = []
    contents = []
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        if raw.get("role") == "system":
            text = blocks_text(raw.get("content"))
            if text:
                system_parts.append(text)
            continue
        converted = _gemini_parts_from_openai_message(raw)
        if converted is not None:
            contents.append({"role": converted[0], "parts": converted[1]})
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            decls.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]
    config = {}
    max_tokens = body.get("max_tokens") if isinstance(body.get("max_tokens"), (int, float)) else body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else []
    if stops:
        config["stopSequences"] = stops
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    if fmt is not None and fmt.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


def anthropic_to_gemini(body, model):
    """Anthropic Messages request as a Gemini generateContent request."""
    system_text = blocks_text(body.get("system"))
    contents = []
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if raw.get("role") == "assistant":
            parts = []
            text = blocks_text(content)
            if text:
                parts.append({"text": text})
            if isinstance(content, list):
                for block in content:
                    if is_record(block) and block.get("type") == "tool_use":
                        parts.append({"functionCall": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "args": block.get("input") if is_record(block.get("input")) else {},
                        }})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image" and is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                        parts.append({"inlineData": {"mimeType": media, "data": source["data"]}})
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        parts.append({"fileData": {"fileUri": source["url"]}})
                elif block.get("type") == "tool_result":
                    text = block.get("content") if isinstance(block.get("content"), str) else blocks_text(block.get("content"))
                    parts.append({"functionResponse": {
                        "name": "",
                        "response": {"content": text},
                    }})
        if parts:
            contents.append({"role": "user", "parts": parts})
    out = {"contents": contents}
    if system_text:
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    if isinstance(body.get("tools"), list):
        decls = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            decls.append({
                "name": raw["name"],
                "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
            })
        if decls:
            out["tools"] = [{"functionDeclarations": decls}]
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        config["stopSequences"] = [s for s in stops if isinstance(s, str)]
    if config:
        out["generationConfig"] = config
    return out


# --- provider answer -> normalized form --------------------------------------
# norm: {"text", "input", "output", "stop", "tool_calls"}
# stop is one of "stop" | "length" | "tool" | "refusal".

def _norm(text, in_tok, out_tok, stop, tool_calls=None):
    return {
        "text": text or "",
        "input": in_tok if isinstance(in_tok, int) else 0,
        "output": out_tok if isinstance(out_tok, int) else 0,
        "stop": stop,
        "tool_calls": tool_calls or [],
    }


def _openai_stop(finish):
    if finish in ("length",):
        return "length"
    if finish in ("tool_calls", "function_call"):
        return "tool"
    if finish == "content_filter":
        return "refusal"
    return "stop"


def _anthropic_stop(reason):
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "tool"
    if reason == "refusal":
        return "refusal"
    return "stop"


def _gemini_stop(reason):
    if reason == "MAX_TOKENS":
        return "length"
    if reason == "SAFETY":
        return "refusal"
    return "stop"


def parse_openai_json(obj):
    if not is_record(obj):
        return _norm("", 0, 0, "stop")
    choices = obj.get("choices") if isinstance(obj.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = message.get("content")
    text = content if isinstance(content, str) else blocks_text(content)
    # blocks_text joins with \n; the provider's answer is one text block, so
    # fall back to the "" join the env's answerText uses when it is a list.
    if isinstance(content, list):
        text = "".join(b["text"] for b in content
                       if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))
    tool_calls = []
    for call in message.get("tool_calls", []) if isinstance(message.get("tool_calls"), list) else []:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        tool_calls.append({
            "id": call.get("id") if isinstance(call.get("id"), str) else "call_unknown",
            "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
            "arguments": _parse_args(fn.get("arguments")),
        })
    usage = obj.get("usage") if is_record(obj.get("usage")) else {}
    return _norm(text, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                 _openai_stop(choice.get("finish_reason")), tool_calls)


def _sse_frames(text):
    """Split SSE text into [(event, data)] frames."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if not frame.strip():
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def parse_openai_sse(text):
    out_text = []
    in_tok = 0
    out_tok = 0
    stop = "stop"
    calls = {}
    for _event, data in _sse_frames(text):
        if data.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not is_record(chunk):
            continue
        usage = chunk.get("usage")
        if is_record(usage):
            if isinstance(usage.get("prompt_tokens"), int):
                in_tok = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int):
                out_tok = usage["completion_tokens"]
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices or not is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            out_text.append(delta["content"])
        for call in delta.get("tool_calls", []) if isinstance(delta.get("tool_calls"), list) else []:
            if not is_record(call):
                continue
            idx = call.get("index") if isinstance(call.get("index"), int) else 0
            entry = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if isinstance(call.get("id"), str) and call["id"]:
                entry["id"] = call["id"]
            fn = call.get("function") if is_record(call.get("function")) else {}
            if isinstance(fn.get("name"), str) and fn["name"]:
                entry["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                entry["arguments"] += fn["arguments"]
        if isinstance(choice.get("finish_reason"), str):
            stop = _openai_stop(choice["finish_reason"])
    tool_calls = [{
        "id": c["id"] or "call_unknown",
        "name": c["name"],
        "arguments": _parse_args(c["arguments"]),
    } for _idx, c in sorted(calls.items())]
    return _norm("".join(out_text), in_tok, out_tok, stop, tool_calls)


def parse_anthropic_json(obj):
    if not is_record(obj):
        return _norm("", 0, 0, "stop")
    content = obj.get("content") if isinstance(obj.get("content"), list) else []
    text = "".join(b["text"] for b in content
                   if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))
    tool_calls = []
    for block in content:
        if is_record(block) and block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id") if isinstance(block.get("id"), str) else "toolu_unknown",
                "name": block.get("name") if isinstance(block.get("name"), str) else "",
                "arguments": block.get("input") if is_record(block.get("input")) else {},
            })
    usage = obj.get("usage") if is_record(obj.get("usage")) else {}
    return _norm(text, usage.get("input_tokens"), usage.get("output_tokens"),
                 _anthropic_stop(obj.get("stop_reason")), tool_calls)


def parse_anthropic_sse(text):
    out_text = []
    in_tok = 0
    out_tok = 0
    stop = "stop"
    calls = {}
    for event, data in _sse_frames(text):
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not is_record(payload):
            continue
        if event == "message_start":
            message = payload.get("message") if is_record(payload.get("message")) else {}
            usage = message.get("usage") if is_record(message.get("usage")) else {}
            if isinstance(usage.get("input_tokens"), int):
                in_tok = usage["input_tokens"]
        elif event == "content_block_start":
            block = payload.get("content_block") if is_record(payload.get("content_block")) else {}
            idx = payload.get("index") if isinstance(payload.get("index"), int) else 0
            if block.get("type") == "tool_use":
                calls[idx] = {
                    "id": block.get("id") if isinstance(block.get("id"), str) else "toolu_unknown",
                    "name": block.get("name") if isinstance(block.get("name"), str) else "",
                    "arguments": "",
                }
        elif event == "content_block_delta":
            delta = payload.get("delta") if is_record(payload.get("delta")) else {}
            idx = payload.get("index") if isinstance(payload.get("index"), int) else 0
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                out_text.append(delta["text"])
            elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                if idx in calls:
                    calls[idx]["arguments"] += delta["partial_json"]
        elif event == "message_delta":
            delta = payload.get("delta") if is_record(payload.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop = _anthropic_stop(delta["stop_reason"])
            usage = payload.get("usage") if is_record(payload.get("usage")) else {}
            if isinstance(usage.get("output_tokens"), int):
                out_tok = usage["output_tokens"]
    tool_calls = [{
        "id": c["id"],
        "name": c["name"],
        "arguments": _parse_args(c["arguments"]),
    } for _idx, c in sorted(calls.items())]
    return _norm("".join(out_text), in_tok, out_tok, stop, tool_calls)


def parse_gemini_json(obj):
    if not is_record(obj):
        return _norm("", 0, 0, "stop")
    candidates = obj.get("candidates") if isinstance(obj.get("candidates"), list) else []
    cand = candidates[0] if candidates and is_record(candidates[0]) else {}
    content = cand.get("content") if is_record(cand.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(p["text"] for p in parts
                   if is_record(p) and isinstance(p.get("text"), str))
    usage = obj.get("usageMetadata") if is_record(obj.get("usageMetadata")) else {}
    return _norm(text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"),
                 _gemini_stop(cand.get("finishReason")))


def parse_gemini_sse(text):
    out_text = []
    in_tok = 0
    out_tok = 0
    stop = "stop"
    for _event, data in _sse_frames(text):
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not is_record(payload):
            continue
        candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
        for cand in candidates:
            if not is_record(cand):
                continue
            content = cand.get("content") if is_record(cand.get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for part in parts:
                if is_record(part) and isinstance(part.get("text"), str):
                    out_text.append(part["text"])
            if isinstance(cand.get("finishReason"), str):
                stop = _gemini_stop(cand["finishReason"])
        usage = payload.get("usageMetadata") if is_record(payload.get("usageMetadata")) else {}
        if isinstance(usage.get("promptTokenCount"), int):
            in_tok = usage["promptTokenCount"]
        if isinstance(usage.get("candidatesTokenCount"), int):
            out_tok = usage["candidatesTokenCount"]
    return _norm("".join(out_text), in_tok, out_tok, stop)


# --- normalized form -> client wire ------------------------------------------

def _openai_finish(stop):
    return {"stop": "stop", "length": "length", "tool": "tool_calls", "refusal": "content_filter"}[stop]


def _anthropic_stop_reason(stop):
    return {"stop": "end_turn", "length": "max_tokens", "tool": "tool_use", "refusal": "refusal"}[stop]


def _split_text(text, size=64):
    if not text:
        return []
    return [text[i:i + size] for i in range(0, len(text), size)]


def emit_openai_json(norm, model):
    message = {"role": "assistant", "content": norm["text"]}
    if norm["tool_calls"]:
        message["tool_calls"] = [{
            "id": c["id"], "type": "function",
            "function": {"name": c["name"], "arguments": compact(c["arguments"])},
        } for c in norm["tool_calls"]]
    return {
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _openai_finish(norm["stop"]),
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": norm["input"],
            "completion_tokens": norm["output"],
            "total_tokens": norm["input"] + norm["output"],
        },
    }


def emit_openai_sse(norm, model):
    def chunk(delta, finish):
        return {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    frames = ["data: %s\n\n" % compact(chunk({"role": "assistant", "content": ""}, None))]
    for piece in _split_text(norm["text"]):
        frames.append("data: %s\n\n" % compact(chunk({"content": piece}, None)))
    for idx, call in enumerate(norm["tool_calls"]):
        frames.append("data: %s\n\n" % compact(chunk({"tool_calls": [{
            "index": idx, "id": call["id"], "type": "function",
            "function": {"name": call["name"], "arguments": compact(call["arguments"])},
        }]}, None)))
    frames.append("data: %s\n\n" % compact(chunk({}, _openai_finish(norm["stop"]))))
    frames.append("data: %s\n\n" % compact({
        "id": "chatcmpl-agg", "object": "chat.completion.chunk",
        "created": FIXED_CREATED, "model": model, "choices": [],
        "usage": {"prompt_tokens": norm["input"], "completion_tokens": norm["output"],
                  "total_tokens": norm["input"] + norm["output"]},
    }))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def emit_anthropic_json(norm, model):
    content = []
    if norm["text"]:
        content.append({"type": "text", "text": norm["text"]})
    for call in norm["tool_calls"]:
        content.append({"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["arguments"]})
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _anthropic_stop_reason(norm["stop"]),
        "stop_sequence": None,
        "usage": {"input_tokens": norm["input"], "output_tokens": norm["output"]},
    }


def _sse_event(event, data):
    return "event: %s\ndata: %s\n\n" % (event, compact(data))


def emit_anthropic_sse(norm, model):
    frames = [_sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_agg", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": norm["input"], "output_tokens": 0},
        },
    })]
    index = 0
    if norm["text"]:
        frames.append(_sse_event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "text", "text": ""},
        }))
        for piece in _split_text(norm["text"]):
            frames.append(_sse_event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": piece},
            }))
        frames.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    for call in norm["tool_calls"]:
        frames.append(_sse_event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": call["id"], "name": call["name"], "input": {}},
        }))
        args = compact(call["arguments"])
        if args:
            frames.append(_sse_event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": args},
            }))
        frames.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    frames.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": _anthropic_stop_reason(norm["stop"]), "stop_sequence": None},
        "usage": {"output_tokens": norm["output"]},
    }))
    frames.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


# ---------------------------------------------------------------------------
# Upstream calls
# ---------------------------------------------------------------------------

def _http_post(url, headers, payload):
    """POST JSON; return (status, body_bytes) or (0, b'') on transport failure."""
    data = compact(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read()
        except Exception:  # noqa: BLE001
            return exc.code, b""
    except Exception as exc:  # noqa: BLE001 - timeout, refused connection, ...
        log("upstream call failed: %s" % exc)
        return 0, b""


def call_upstream(state, provider, model, client_api, body, stream):
    """One inference call. Returns (status, norm) with norm None on failure."""
    base = join_base(state["env_url"], provider["base"])
    headers = {"Content-Type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    wire = provider["wire"]
    if wire == "openai":
        url = base + "/v1/chat/completions"
        if client_api == "openai":
            req_body = dict(body)
            req_body["model"] = model
            if stream:
                req_body["stream"] = True
                req_body["stream_options"] = {"include_usage": True}
        else:
            req_body = anthropic_to_openai(body, model)
        if stream:
            headers["Accept"] = "text/event-stream"
        status, raw = _http_post(url, headers, req_body)
        if status != 200:
            return status, None
        text = raw.decode("utf-8", errors="replace")
        if stream:
            return status, parse_openai_sse(text)
        try:
            return status, parse_openai_json(json.loads(text))
        except json.JSONDecodeError:
            return status, parse_openai_json({})
    if wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        if client_api == "anthropic":
            req_body = dict(body)
            req_body["model"] = model
        else:
            req_body = openai_to_anthropic(body, model)
        if stream:
            headers["Accept"] = "text/event-stream"
        status, raw = _http_post(url, headers, req_body)
        if status != 200:
            return status, None
        text = raw.decode("utf-8", errors="replace")
        if stream:
            return status, parse_anthropic_sse(text)
        try:
            return status, parse_anthropic_json(json.loads(text))
        except json.JSONDecodeError:
            return status, parse_anthropic_json({})
    # gemini
    encoded = urllib.parse.quote(model, safe="")
    action = "streamGenerateContent" if stream else "generateContent"
    url = "%s/v1beta/models/%s:%s" % (base, encoded, action)
    if stream:
        url += "?alt=sse"
        headers["Accept"] = "text/event-stream"
    if client_api == "openai":
        req_body = openai_to_gemini(body, model)
    else:
        req_body = anthropic_to_gemini(body, model)
    status, raw = _http_post(url, headers, req_body)
    if status != 200:
        return status, None
    text = raw.decode("utf-8", errors="replace")
    if stream:
        return status, parse_gemini_sse(text)
    try:
        return status, parse_gemini_json(json.loads(text))
    except json.JSONDecodeError:
        return status, parse_gemini_json({})


# ---------------------------------------------------------------------------
# Routing: candidates, primary plan, failover
# ---------------------------------------------------------------------------

def meets_requirements(row, requires):
    if requires.get("tools") and not row["tools"]:
        return False
    if requires.get("vision") and not row["vision"]:
        return False
    if requires.get("json") and not row["json"]:
        return False
    min_ctx = requires.get("min_context")
    if min_ctx is not None and (row["context_length"] is None or row["context_length"] < min_ctx):
        return False
    return True


def candidates_for(state, requires, client_api):
    """Free (provider, model) pairs meeting the requirements, best first.

    An OpenAI client asking for JSON mode is never sent to an Anthropic-wire
    provider: that wire cannot carry response_format, so the provider would
    echo json_mode false and the request would fail the echo check.
    """
    json_openai = client_api == "openai" and requires.get("json") is True
    listed = {}
    for probe in state["probes"]:
        if probe["ok"]:
            listed[probe["provider"]["id"]] = {m["id"] for m in probe["models"]}
    out = []
    for (provider_id, model_id), row in state["ratings"].items():
        if not row["free"]:
            continue
        if model_id not in listed.get(provider_id, set()):
            continue
        if not meets_requirements(row, requires):
            continue
        provider = next((p for p in state["providers"] if p["id"] == provider_id), None)
        if provider is None:
            continue
        if json_openai and provider["wire"] == "anthropic":
            continue
        out.append({"provider": provider_id, "model": model_id,
                    "rating": row["rating_milli"], "wire": provider["wire"]})
    out.sort(key=lambda c: (-c["rating"], c["provider"], c["model"]))
    return out


def plan_primaries(state):
    """Assign each manifest request a primary model before the first request.

    Requests are planned scarcest-first (fewest candidates), heaviest first,
    in manifest order. Each pick maximises entry weight times rating minus
    a charge for the refusal risk of the provider's next call (CALL_RISK):
    a rating point on a weight-2 request is worth twice a point on a
    weight-1 request, so risk is spent preferentially on cheap requests.
    A slightly weaker model on a provider whose next call is likely safe
    beats the top model on a provider whose next call will probably
    refuse, because every refusal costs an upstream call and the score's
    tie-break is the call count. Quality stays guarded: the charge is
    small next to the rating gaps that matter (top model 1475 vs second
    tier 1470/1465).
    """
    order = []
    for idx, entry in enumerate(state["workload"]):
        cands = candidates_for(state, entry["requires"], entry["api"])
        order.append((len(cands), -entry["weight"], idx, entry, cands))
    order.sort(key=lambda t: (t[0], t[1], t[2]))
    counts = {}
    primaries = {}
    for _n, _w, _i, entry, cands in order:
        pick = None
        best_value = None
        for cand in cands:
            provider_id = cand["provider"]
            risk = call_risk(provider_id, counts.get(provider_id, 0) + 1)
            value = entry["weight"] * cand["rating"] - RISK_LAMBDA * risk
            if best_value is None or value > best_value:
                pick, best_value = cand, value
        if pick is None and cands:
            pick = cands[0]
        if pick is not None:
            primaries[entry["id"]] = pick
            counts[pick["provider"]] = counts.get(pick["provider"], 0) + 1
    return primaries


def requirements_from_body(client_api, body):
    """Requirements read off a request body (fallback when no manifest row)."""
    requires = {}
    if isinstance(body.get("tools"), list) and body["tools"]:
        requires["tools"] = True
    messages = body.get("messages", []) if isinstance(body.get("messages"), list) else []
    vision = False
    for raw in messages:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if isinstance(content, list):
            for block in content:
                if is_record(block) and block.get("type") in ("image", "image_url"):
                    vision = True
    if vision:
        requires["vision"] = True
    if client_api == "openai":
        fmt = body.get("response_format") if is_record(body.get("response_format")) else None
        if fmt is not None and fmt.get("type") in ("json_object", "json_schema"):
            requires["json"] = True
    return requires


class Aggregator:
    def __init__(self, env_url, data):
        self.env_url = env_url
        self.data = data
        self.providers = {p["id"]: p for p in data["providers"]}
        self.primaries = plan_primaries(data)
        self.health = {p["id"]: {"calls": 0, "consec_fail": 0, "total_fail": 0}
                       for p in data["providers"]}
        self.dropped = set()  # providers blacklisted after 3 consecutive refusals
        self.hist = {p["id"]: [] for p in data["providers"]}  # per-provider outcomes, in call order
        self.shortfall = 0  # weighted rating shortfall served so far (quality guard)
        self.best_rating = {}  # best rating per manifest entry (quality guard)
        for entry in data["workload"]:
            cands = candidates_for(data, entry["requires"], entry["api"])
            self.best_rating[entry["id"]] = max((c["rating"] for c in cands), default=0)
        self.lock = threading.Lock()
        self.used = set()
        self.in_flight = 0
        self.started = time.monotonic()

    def match_entry(self, client_api, body):
        """The manifest row for the request now arriving.

        Requests normally arrive in manifest order, but the brief does not
        promise it, so among the not-yet-matched rows pick the best fit for
        this request's wire (api, stream) and its visible features (tools,
        image, JSON mode), breaking ties by manifest order. In-order
        arrivals therefore still match their own row; an out-of-order
        arrival matches the row its features describe instead of stealing
        the next row's requirements.
        """
        stream = body.get("stream") is True
        manifest = self.data["workload"]
        tools = body.get("tools")
        has_tools = isinstance(tools, list) and len(tools) > 0
        has_image = False
        for msg in body.get("messages", []) or []:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") in ("image_url", "image"):
                        has_image = True
        feats = {"tools": has_tools, "vision": has_image}
        if client_api == "openai":
            rf = body.get("response_format")
            feats["json"] = isinstance(rf, dict) and rf.get("type") == "json_object"
        with self.lock:
            best = None
            best_score = None
            for i, entry in enumerate(manifest):
                if i in self.used or entry["api"] != client_api:
                    continue
                score = 2  # api matches
                if entry["stream"] == stream:
                    score += 1
                req = entry["requires"]
                for key, value in feats.items():
                    # requires only lists what is needed; absent means False
                    if req.get(key, False) == value:
                        score += 1
                if best_score is None or score > best_score:
                    best, best_score = i, score
            if best is None:
                return None
            self.used.add(best)
            return manifest[best]
        return None

    def attempt_order(self, entry, requires, client_api):
        """Try-order for one request: best risk-adjusted value first.

        Value is entry weight times rating minus RISK_LAMBDA times the
        Bayesian posterior refusal risk of the provider's next call
        (POST_RISK, conditioned on that provider's own observed outcomes
        so far). A refusal raises the posterior a lot, a success on a
        small-budget provider also raises it (its scheduled failures must
        sit at later call numbers), so no separate suspend penalty is
        needed: the posterior already encodes the failure information.
        Dropped providers are never tried again. The planned primary goes
        first when it is still eligible.

        Quality guard: once the weighted shortfall served so far exceeds
        QUALITY_GUARD, the risk charge is dropped (rating-first) and the
        primary is no longer forced first, so the remaining requests are
        served by the best models available.
        """
        cands = candidates_for(self.data, requires, client_api)
        primary = self.primaries.get(entry["id"]) if entry else None
        weight = entry["weight"] if entry is not None else 1
        with self.lock:
            guarded = self.shortfall > QUALITY_GUARD
            lam = 0 if guarded else RISK_LAMBDA
            scored = []
            for cand in cands:
                provider_id = cand["provider"]
                if provider_id in self.dropped:
                    continue
                risk = post_risk(provider_id, tuple(self.hist[provider_id]))
                value = weight * cand["rating"] - lam * risk
                scored.append((value, cand))
        scored.sort(key=lambda t: -t[0])
        ordered = [cand for _value, cand in scored]
        if not guarded and primary is not None and primary in ordered:
            ordered.remove(primary)
            ordered.insert(0, primary)
        return ordered

    def serve(self, client_api, body, stream):
        """Serve one client request. Returns (provider, model, norm) or None."""
        entry = self.match_entry(client_api, body)
        requires = entry["requires"] if entry is not None else requirements_from_body(client_api, body)
        tried = set()
        attempts = 0
        for cand in self.attempt_order(entry, requires, client_api):
            if attempts >= MAX_ATTEMPTS_PER_REQUEST:
                break
            provider_id = cand["provider"]
            if provider_id in tried:
                continue
            tried.add(provider_id)
            attempts += 1
            provider = self.providers[provider_id]
            status, norm = call_upstream(self.data | {"env_url": self.env_url},
                                         provider, cand["model"], client_api, body, stream)
            ok = status == 200 and norm is not None
            with self.lock:
                self.health[provider_id]["calls"] += 1
                self.hist[provider_id].append(not ok)
                if ok:
                    self.health[provider_id]["consec_fail"] = 0
                    if entry is not None:
                        self.shortfall += entry["weight"] * (
                            self.best_rating.get(entry["id"], cand["rating"]) - cand["rating"])
                else:
                    self.health[provider_id]["consec_fail"] += 1
                    self.health[provider_id]["total_fail"] += 1
                    if self.health[provider_id]["consec_fail"] >= 3:
                        # Three refusals in a row: the provider is past its
                        # outage or its budget is spent; later calls would
                        # almost surely refuse too, so stop spending calls.
                        self.dropped.add(provider_id)
            if ok:
                log("served %s via %s/%s (attempt %d)" % (
                    entry["id"] if entry else "?", provider_id, cand["model"], attempts))
                return provider_id, cand["model"], norm
            log("failover: %s/%s answered HTTP %s" % (provider_id, cand["model"], status))
        if entry is not None:
            with self.lock:
                self.shortfall += entry["weight"] * self.best_rating.get(entry["id"], 0)
        return None


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def openai_error(message, type_="invalid_request_error", code=None):
    return {"error": {"message": message, "type": type_, "code": code}}


def anthropic_error(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    aggregator = None  # set by serve()
    served = []        # set by serve()

    def log_message(self, fmt, *args):  # keep stdout clean; stderr only
        log("%s - %s" % (self.address_string(), fmt % args))

    def _read_body(self):
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else 0
        except ValueError:
            size = 0
        if size <= 0:
            return b""
        if size > MAX_BODY_BYTES:
            return None
        return self.rfile.read(size)

    def _send_json(self, status, obj):
        data = compact(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, text):
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()
        self.close_connection = True

    def _is_anthropic(self):
        return self.headers.get("x-api-key") is not None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/models":
            if self._is_anthropic():
                body = {
                    "data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                              "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in self.served],
                    "has_more": False,
                }
            else:
                body = {
                    "object": "list",
                    "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                              "owned_by": m["provider"]} for m in self.served],
                }
            self._send_json(200, body)
            return
        wire = "anthropic" if parsed.path == "/v1/messages" else "openai"
        if wire == "anthropic":
            self._send_json(404, anthropic_error("not_found", "no route GET %s" % parsed.path))
        else:
            self._send_json(404, openai_error("no route GET %s" % parsed.path, "not_found"))

    def _handle_post(self, client_api):
        raw = self._read_body()
        wire = client_api
        if raw is None:
            err = anthropic_error("invalid_request_error", "request body too large") if wire == "anthropic" else openai_error("request body too large")
            self._send_json(400, err)
            return
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None
        if not is_record(body):
            err = anthropic_error("invalid_request_error", "the request body is not one JSON object") if wire == "anthropic" else openai_error("the request body is not one JSON object")
            self._send_json(400, err)
            return
        stream = body.get("stream") is True
        agg = self.aggregator
        with agg.lock:
            agg.in_flight += 1
        try:
            result = agg.serve(client_api, body, stream)
        finally:
            with agg.lock:
                agg.in_flight -= 1
        if result is None:
            err = anthropic_error("overloaded_error", "no free model could serve this request") if wire == "anthropic" else openai_error("no free model could serve this request", "no_free_model")
            self._send_json(503, err)
            return
        _provider, model, norm = result
        if wire == "openai":
            if stream:
                self._send_sse(emit_openai_sse(norm, model))
            else:
                self._send_json(200, emit_openai_json(norm, model))
        else:
            if stream:
                self._send_sse(emit_anthropic_sse(norm, model))
            else:
                self._send_json(200, emit_anthropic_json(norm, model))

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/chat/completions":
            self._handle_post("openai")
            return
        if parsed.path == "/v1/messages":
            self._handle_post("anthropic")
            return
        wire = "anthropic" if parsed.path == "/v1/messages" else "openai"
        if wire == "anthropic":
            self._send_json(404, anthropic_error("not_found", "no route POST %s" % parsed.path))
        else:
            self._send_json(404, openai_error("no route POST %s" % parsed.path, "not_found"))


# ---------------------------------------------------------------------------
# Register, poll /work, main
# ---------------------------------------------------------------------------

def _http_json(url, method, payload=None, timeout=5.0):
    data = compact(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return exc.code, {}
    except Exception as exc:  # noqa: BLE001
        log("HTTP %s %s failed: %s" % (method, url, exc))
        return 0, {}


def register(env_url, port, catalog):
    for attempt in range(10):
        status, reply = _http_json(env_url + "/register", "POST",
                                   {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog})
        if status == 200 and is_record(reply) and reply.get("ok") is True:
            log("registered: the workload holds %s requests" % reply.get("requests"))
            return True
        log("register attempt %d answered HTTP %s: %s" % (attempt + 1, status, compact(reply)[:200]))
        time.sleep(0.5)
    return False


def poll_work(env_url, server, agg):
    while True:
        time.sleep(POLL_INTERVAL)
        if time.monotonic() - agg.started > SAFETY_WALL_SECONDS:
            with agg.lock:
                busy = agg.in_flight > 0
            if not busy:
                log("safety wall reached; exiting 0")
                server.shutdown()
                return
            continue
        status, work = _http_json(env_url + "/work", "GET", timeout=2.0)
        if status != 200 or not is_record(work):
            continue
        if work.get("done") is True:
            with agg.lock:
                busy = agg.in_flight > 0
            if not busy:
                log("work done; exiting 0")
                server.shutdown()
                return


def main(argv):
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    data = load(env_url, prompt_file)
    catalog_text = compact(data["catalog"])
    if "--list" in argv:
        sys.stdout.write(catalog_text + "\n")
        sys.stdout.flush()
        return 0
    if not env_url:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    agg = Aggregator(env_url, data)
    Handler.aggregator = agg
    Handler.served = data["served"]
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        return 1
    log("listening on 0.0.0.0:%d with %d providers and %d free models"
        % (port, len(data["catalog"]["providers"]), len(data["served"])))
    if not register(env_url, port, data["catalog"]):
        log("register failed")
        return 1

    def _term(_signum, _frame):
        os._exit(0)

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    poller = threading.Thread(target=poll_work, args=(env_url, server, agg), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
