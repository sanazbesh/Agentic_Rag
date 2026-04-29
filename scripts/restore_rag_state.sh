#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="docker-compose.yml"
BACKUP_DIR="${1:-}"
POSTGRES_CONTAINER="agentic_rag_postgres"
QDRANT_CONTAINER="agentic_rag_qdrant"
APP_CONTAINER="agentic_rag_app"

if [[ -z "$BACKUP_DIR" ]]; then
  echo "Usage: $0 <backup_dir>" >&2
  exit 1
fi

if [[ ! -d "$BACKUP_DIR" ]]; then
  echo "Error: backup directory does not exist: $BACKUP_DIR" >&2
  exit 1
fi

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

ensure_running postgres
ensure_running qdrant
ensure_running app

if [[ ! -f "$BACKUP_DIR/postgres/agentic_rag.dump" ]]; then
  echo "Error: missing Postgres backup file: $BACKUP_DIR/postgres/agentic_rag.dump" >&2
  exit 1
fi

if [[ ! -f "$BACKUP_DIR/documents/documents.tgz" ]]; then
  echo "Error: missing documents backup file: $BACKUP_DIR/documents/documents.tgz" >&2
  exit 1
fi

echo "[1/3] Restoring Postgres..."
docker exec "$POSTGRES_CONTAINER" sh -lc 'psql -U agentic_rag -d agentic_rag -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"'
docker exec -i "$POSTGRES_CONTAINER" sh -lc 'pg_restore -U agentic_rag -d agentic_rag --clean --if-exists --no-owner --no-privileges' < "$BACKUP_DIR/postgres/agentic_rag.dump"

echo "[2/3] Restoring Qdrant..."
if [[ -f "$BACKUP_DIR/qdrant/snapshots_manifest.txt" ]]; then
  while IFS=':' read -r collection snapshot_file; do
    [[ -z "$collection" || -z "$snapshot_file" ]] && continue
    snapshot_path="$BACKUP_DIR/qdrant/$snapshot_file"
    if [[ ! -f "$snapshot_path" ]]; then
      echo "Warning: missing snapshot file for collection '$collection': $snapshot_path" >&2
      continue
    fi
    curl -fsS -X PUT "http://localhost:6333/collections/${collection}/snapshots/upload" \
      -H 'Content-Type: multipart/form-data' \
      -F "snapshot=@${snapshot_path}" > /dev/null
  done < "$BACKUP_DIR/qdrant/snapshots_manifest.txt"
elif [[ -f "$BACKUP_DIR/qdrant/storage_fallback.tgz" ]]; then
  docker exec "$QDRANT_CONTAINER" sh -lc 'rm -rf /qdrant/storage/*'
  docker exec -i "$QDRANT_CONTAINER" sh -lc 'tar -C /qdrant/storage -xzf -' < "$BACKUP_DIR/qdrant/storage_fallback.tgz"
  docker restart "$QDRANT_CONTAINER" >/dev/null
else
  echo "Error: no Qdrant snapshot manifest or storage fallback found in backup." >&2
  exit 1
fi

echo "[3/3] Restoring documents..."
docker exec "$APP_CONTAINER" sh -lc 'mkdir -p /app/data/documents && rm -rf /app/data/documents/*'
docker exec -i "$APP_CONTAINER" sh -lc 'tar -C /app/data/documents -xzf -' < "$BACKUP_DIR/documents/documents.tgz"

echo "Restore complete from: $BACKUP_DIR"
