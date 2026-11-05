#!/usr/bin/env bash
# A session is a stretch with the LLM pod running.
#   session.sh start BASE_URL   point the API at it, keep one replica warm, wait for /ready
#   session.sh stop             clear the URL and let the API scale back to zero
# Both refresh the UI's allowlist, so a changed home IP doesn't lock you out.
source "$(dirname "$0")/common.sh"

case "${1:-}" in
    start)
        llm_url="${2:-}"
        [ -n "$llm_url" ] || die "usage: session.sh start BASE_URL"
        min_replicas=1
        ;;
    stop)
        llm_url="$NO_LLM"
        min_replicas=0
        ;;
    *) die "usage: session.sh start BASE_URL | session.sh stop" ;;
esac

api="$(output apiAppName)"
ui="$(output uiAppName)"
ip="$(public_ip)"

az containerapp ingress access-restriction set -n "$ui" -g "$RG" --rule-name allow-one-ip \
    --ip-address "$ip/32" --action Allow --output none
az containerapp update -n "$api" -g "$RG" --container-name api \
    --set-env-vars "LLM_BASE_URL=$llm_url" --min-replicas "$min_replicas" --max-replicas 1 \
    --output none

[ "$1" = start ] || { echo "session stopped"; exit 0; }

fqdn="$(az containerapp show -n "$ui" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
code=000
for _ in $(seq 60); do  # 10 minutes: the new revision copies its data first
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$fqdn/api/ready" || true)"
    if [ "$code" = 200 ]; then
        echo "session started: https://$fqdn"
        exit 0
    fi
    sleep 10
done
die "/api/ready still answers $code after 10 minutes"
