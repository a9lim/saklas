# Security Policy

## Reporting a vulnerability

If you've found a security issue in saklas, please report it privately rather than filing a public issue.

- **Email:** mx@a9l.im
- **GitHub:** use [private security advisories](https://github.com/a9lim/saklas/security/advisories/new)

Include a description, reproduction steps, and the saklas version. I'll acknowledge within a few days and aim to have a fix or mitigation out before any public disclosure.

## Supported versions

Only the latest minor release on PyPI receives security fixes. If you're on an older version, the fix is to upgrade.

## Threat model for `saklas serve`

The OpenAI-compatible API server (`saklas serve`) is designed for **trusted networks** — a local dev machine, a lab VPN, or a single-tenant container. It is **not** hardened for direct exposure to the public internet.

What the server does provide:

- Optional bearer auth via `--api-key` or `$SAKLAS_API_KEY`. If unset, the server is open.
- Per-request serialization through a single `asyncio.Lock`, so one slow generation can't interleave with another.
- Request validation via pydantic for all sampling parameters.

What it does not provide:

- Rate limiting
- Per-user quotas or isolation
- Protection against adversarial `logit_bias` / `stop` / `max_tokens` inputs designed to slow generation
- TLS (run it behind a reverse proxy if you need HTTPS)
- Any kind of sandboxing for the loaded model

If you need to expose saklas to untrusted callers, put it behind a reverse proxy (nginx, Caddy, Cloudflare Tunnel) with its own auth, rate limiting, and request size limits.

## Model and checkpoint trust

saklas loads HuggingFace checkpoints via `transformers`, which executes code from the checkpoint repo in some cases (custom modeling code, `trust_remote_code=True`). saklas does not set `trust_remote_code=True` by default, but if you pass a model that requires it, be aware you are executing arbitrary code from that repo. Only load models from publishers you trust.

Steering vector packs pulled from HuggingFace (`saklas -i <owner>/<name>`) are verified against the `files` sha256 map in `pack.json`, so on-disk tampering after download is detected. They do **not** have publisher signatures yet — pack signing is reserved for a future version (see `docs/superpowers/specs/2026-04-12-story-a-portability-design.md`). For now, only install packs from publishers you trust.
