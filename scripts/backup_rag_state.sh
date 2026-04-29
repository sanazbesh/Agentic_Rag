#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="docker-compose.yml"
BACKUP_ROOT="${1:-backups}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
BACKUP_DIR="${BACKUP_ROOT%/}/${TIMESTAMP}"
POSTGRES_CONTAINER="agentic_rag_postgres"
QDRANT_CONTAINER="agentic_rag_qdrant"
APP_CONTAINER="agentic_rag_app"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: required command '$1' is not installed." >&2
    exit 1
  }
}

ensure_running() {
  local svc="$1"
  local cid
  cid="$(docker compose -f "$COMPOSE_FILE" ps -q "$svc")"
  if [[ -z "$cid" ]]; then
    echo "Error: service '$svc' is not running. Start stack with: docker compose up -d" >&2
    exit 1
  fi
}

require_cmd docker
require_cmd curl
require_cmd tar

ensure_running postgres
ensure_running qdrant
ensure_running app

mkdir -p "$BACKUP_ROOT"
if [[ -e "$BACKUP_DIR" ]]; then
  echo "Error: backup directory already exists: $BACKUP_DIR" >&2
  exit 1
fi
mkdir -p "$BACKUP_DIR" "$BACKUP_DIR/postgres" "$BACKUP_DIR/qdrant" "$BACKUP_DIR/documents"

echo "[1/3] Backing up Postgres..."
docker exec "$POSTGRES_CONTAINER" sh -lc 'pg_dump -U agentic_rag -d agentic_rag -F c' > "$BACKUP_DIR/postgres/agentic_rag.dump"

echo "[2/3] Backing up Qdrant..."
collections_json="$(curl -fsS http://localhost:6333/collections)"
collection_names="$(printf '%s' "$collections_json" | sed -n 's/.*"name":"\([^"]*\)".*/\1/p')"
if [[ -n "$collection_names" ]]; then
  while IFS= read -r collection; do
    [[ -z "$collection" ]] && continue
    snapshot_name="backup_${TIMESTAMP}_${collection}.snapshot"
    curl -fsS -X POST "http://localhost:6333/collections/${collection}/snapshots" \
      -H 'Content-Type: application/json' > /tmp/qdrant_snapshot_resp.json
    created_name="$(sed -n 's/.*"name":"\([^"]*\)".*/\1/p' /tmp/qdrant_snapshot_resp.json | head -n1)"
    if [[ -n "$created_name" ]]; then
      curl -fsS "http://localhost:6333/collections/${collection}/snapshots/${created_name}" -o "$BACKUP_DIR/qdrant/${snapshot_name}"
      printf '%s\n' "$collection:$snapshot_name" >> "$BACKUP_DIR/qdrant/snapshots_manifest.txt"
    fi
  done <<< "$collection_names"
fi
rm -f /tmp/qdrant_snapshot_resp.json

if [[ ! -f "$BACKUP_DIR/qdrant/snapshots_manifest.txt" ]]; then
  echo "No Qdrant snapshots created; exporting raw storage volume as fallback."
  docker exec "$QDRANT_CONTAINER" sh -lc 'tar -C /qdrant/storage -czf - .' > "$BACKUP_DIR/qdrant/storage_fallback.tgz"
fi

echo "[3/3] Backing up document storage..."
docker exec "$APP_CONTAINER" sh -lc 'tar -C /app/data/documents -czf - .' > "$BACKUP_DIR/documents/documents.tgz"

echo "Backup complete: $BACKUP_DIR"
