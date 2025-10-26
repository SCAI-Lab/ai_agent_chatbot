# Project Context

## Purpose
Memobase delivers persistent, user-centric memory for LLM applications. It ingests chat transcripts, curates long-lived user profiles and event timelines, and exposes those memories through APIs and SDKs so assistants can personalize responses, track preferences, and evolve with each user while keeping latency and token usage low.

## Tech Stack
- Python 3.12 + FastAPI for the primary API server
- PostgreSQL (with pgvector) as the durable memory store and Redis as the low-latency cache/buffer layer
- SQLAlchemy, Alembic, and structlog/OpenTelemetry for data access, migrations, and observability
- OpenAI-compatible LLM/embedding providers (OpenAI, Volc Ark, Jina) orchestrated via configurable prompts
- First-party SDKs in Python (pip/uv), TypeScript (npm/jsr), Go (Go modules), and MCP for agent integrations
- Docker Compose + uv for local development, packaging, and environment provisioning

## Project Conventions

### Code Style
Python services follow PEP 8 with comprehensive type hints; formatters/lints default to `black` and `ruff` (via `uv run`) before commits. TypeScript code targets modern ES modules with strict `tsconfig` settings, Jest for tests, and ESLint/Prettier defaults. Go modules adhere to standard Go formatting (`gofmt`, `goimports`) and idiomatic package naming.

### Architecture Patterns
Server code uses a layered FastAPI architecture: route definitions in `api.py`, controllers encapsulating business logic, typed models (SQLAlchemy + Pydantic) for persistence/IO, and connector abstractions for Postgres/Redis/LLM providers. Memory processing pipelines flush chat buffers into structured profiles and event gists, keeping long-term storage and retrieval isolated from request handlers. Client SDKs wrap HTTP APIs with thin language-specific abstractions while reusing shared schema conventions.

### Testing Strategy
Primary server tests run with `uv run pytest`, combining async FastAPI integration tests, controller-level units, and migration checks (`pytest-asyncio`, `pytest-cov`). Client SDKs ship language-appropriate suites (`pytest` for Python, `jest` for TypeScript, `go test ./...` for Go) that validate request builders, serializers, and golden responses. Changes should expand coverage around new memory flows (buffer flush, profile generation) and include regression cases for critical APIs such as `context`, `profile`, and `buffer` endpoints.

### Git Workflow
Work branches fork from `dev` using `feature/*` or `fix/*` naming. Commits follow Conventional Commit headers (e.g., `feat:`, `fix:`) and get rebased on `dev` before PR submission. PRs must pass automated checks, include relevant tests/documentation updates, and keep histories clean via interactive rebase/squash.

## Domain Context
Key domain entities are Users, Profiles (hierarchical topics/subtopics describing preferences, traits, and context), Events (time-ordered chat gists with metadata), and Buffers (short-lived chat accumulation). The system emphasizes long-term personalization: chats are batched, summarized with LLM prompts, embedded for retrieval, and surfaced via `/context` and `/profile` APIs. Understanding how profile slots map to downstream prompts and how event filters drive recommendation logic is critical when extending features.

## Important Constraints
- Maintain sub-100 ms latency for hot paths by leaning on Redis caches and efficient SQL queries; avoid heavy synchronous LLM calls inside request handlers.
- Respect token and cost budgets by batching buffer flushes and reusing prompts; new flows must not increase the fixed three-call LLM pipeline without review.
- Secure handling of secrets (LLM/embedding keys) and tenant data is mandatory; configuration supports env overrides and should not leak credentials.
- Deployments assume managed Postgres and Redis instances; schema/order changes require corresponding Alembic migrations and backward compatibility planning.

## External Dependencies
- PostgreSQL 15+ with `pgvector` extension
- Redis 7+ for caching, rate limiting, and buffer queues
- OpenAI-compatible chat and embedding APIs (OpenAI, Volc Ark, local vLLM/Ollama endpoints, Jina embeddings)
- Docker/Docker Compose for local orchestration and CI parity
- Telemetry sinks (Prometheus via OpenTelemetry exporter) for metrics and tracing
