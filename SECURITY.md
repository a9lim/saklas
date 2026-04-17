# Security Policy

## Reporting a vulnerability

If you've found a security issue in saklas, please report it privately rather than filing a public issue.

- **Email:** mx@a9l.im
- **GitHub:** use [private security advisories](https://github.com/a9lim/saklas/security/advisories/new)

Please include a description, steps to reproduce, and the version you are on. I'll respond within a few days and aim to have a fix as soon as possible.

## Supported versions

Only the latest minor version on PyPI receives security fixes. If you're on an older version, the fix is to upgrade.

## Threat model for `saklas serve`

The HTTP server (`saklas serve`) is meant to be used on a trusted network, like a local machine or a lab VPN. It's not meant to be used on the public internet. Please use caution.

What it does:

- Optional bearer auth via `--api-key` or `$SAKLAS_API_KEY`. If unset, the server is open.
- Per-request serialization through a single `asyncio.Lock`, so one slow generation doesn't interleave with another.
- Request validation via pydantic for all sampling parameters.

What it does not do:

- Rate limit
- User quotas or isolation
- Protection against adversarial inputs (`logit_bias`, `stop`, `max_tokens` designed to slow generation)
- TLS (please run it behind a reverse proxy if you need HTTPS)
- Sandboxing for the loaded model

If you need to expose saklas to callers you don't trust, please use a reverse proxy.

## Model and checkpoint trust

Saklas loads HuggingFace checkpoints via `transformers`, which executes code from the checkpoint repo in some cases. saklas does not set `trust_remote_code=True` by default, but if you pass a model that requires it, please be aware that you are executing arbitrary code from that repo, so only load models from publishers you trust.

Steering vector packs pulled from HuggingFace (`saklas pack install <owner>/<name>`) are verified against the `files` sha256 map in `pack.json`, so on-disk tampering after download is detected. Please only install packs from publishers you trust.
