#!/usr/bin/env bash
# Copy what the apps start from into private Blob: the index, the manuals and a Qdrant snapshot
# taken from the local stack (make up). Safe to rerun; every blob is overwritten.
source "$(dirname "$0")/common.sh"

QDRANT="http://localhost:6333"
COLLECTION="rig_assist_chunks"  # the Qdrant app restores this name; see apps.bicep

account="$(output storageAccountName)"
container="$(output dataContainer)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

upload() {
    az storage blob upload-batch --auth-mode login --account-name "$account" \
        --destination "$container" --source "$1" --destination-path "$2" \
        --overwrite --only-show-errors --output none
}

snapshot="$(curl -fsS -X POST "$QDRANT/collections/$COLLECTION/snapshots" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["result"]["name"])')"
curl -fsS "$QDRANT/collections/$COLLECTION/snapshots/$snapshot" -o "$tmp/$COLLECTION.snapshot"
curl -fsS -X DELETE "$QDRANT/collections/$COLLECTION/snapshots/$snapshot" -o /dev/null

# Only what the API reads: data/index also holds eval baselines and parser variants.
upload data/index/chunks index/chunks
upload data/index/documents index/documents
upload data/raw raw
upload "$tmp" qdrant

for prefix in index/ raw/ qdrant/; do
    count="$(az storage blob list --auth-mode login --account-name "$account" \
        --container-name "$container" --prefix "$prefix" --query 'length(@)' -o tsv)"
    echo "$prefix: $count blobs"
done
