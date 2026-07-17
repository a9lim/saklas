# Security Policy

## Reporting a vulnerability

If you've found a security issue in saklas, please report it privately rather than filing a public issue.

- **Email:** mx@a9l.im
- **GitHub:** use [private security advisories](https://github.com/a9lim/saklas/security/advisories/new)

Please include a description, reproduction steps, affected version, model ID, and
whether the server was reachable beyond localhost. Do not include API keys,
private prompts, or credentials in the report.

## Supported versions

Only the latest release on PyPI receives security fixes. Upgrade before reporting
an issue that may already be fixed.

## Threat model for `saklas serve`

The HTTP server (`saklas serve`) is designed for a single trusted user on a local
machine or trusted lab network. It is not a hardened multi-tenant service and
should not be exposed directly to the public internet.

What it does:

- Optional bearer auth via `--api-key` or `$SAKLAS_API_KEY`. If unset, every HTTP
  and WebSocket route is open.
- A bounded async session lock serializes generation-facing OpenAI, Ollama, and
  native requests before they enter the engine; the synchronous session also
  rejects generation re-entry.
- Pydantic validates protocol request bodies; native request models reject unknown
  fields.
- Installed manifold payloads and Saklas-owned fitted artifacts are checked
  against their declared SHA-256 digests before use; external J-lens/SAE sources
  are commit- or release-pinned through local bindings.

What it does not do:

- Rate limiting, quotas, per-user isolation, or audit logging
- Resource isolation from deliberately expensive prompts, sampling options,
  fitting jobs, or repeated downloads
- TLS; use a correctly configured reverse proxy if HTTPS is required
- Sandboxing for model code, tokenizers, checkpoints, or downloaded artifacts
- Protection against a bearer token appearing in browser WebSocket query strings
  and intermediary logs (the bundled dashboard uses `?token=` because browser
  WebSocket APIs cannot set an `Authorization` header)

If untrusted callers need access, add authentication, TLS, request/body limits,
rate limits, and process-level isolation outside Saklas. A reverse proxy alone is
not a complete isolation boundary.

## Model and checkpoint trust

Saklas resolves Hugging Face configuration and tokenizer metadata with remote-code
support enabled, then avoids custom model code when the architecture is supported
natively by Transformers. Repositories that are not natively supported may execute
their model implementation as well. Treat every model repository as executable
code and load only revisions and publishers you trust.

`saklas pack install <owner>/<name>` verifies files declared by the manifold's
`manifold.json` integrity map, but integrity is not authorship or safety. Manifold
metadata and corpora remain untrusted input; install only from publishers you
trust. Provider-owned J-lens and SAE payloads remain in their provider cache and
are pinned through local bindings.
