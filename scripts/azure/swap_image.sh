#!/usr/bin/env bash
# swap_image.sh APP CONTAINER IMAGE: deploy a new image and wait until it has passed its probes.
# At zero replicas a revision can report success without ever starting, so the app keeps one
# replica until the revision is healthy, then goes back to its previous setting. Used by
# deploy.yml: its log is public, so this prints states only, never app logs or az's own output
# (nginx logs and the app's config hold the allowed IP).
source "$(dirname "$0")/common.sh"

app="$1"
container="$2"
image="$3"
min_before="$(az containerapp show -n "$app" -g "$RG" \
    --query properties.template.scale.minReplicas -o tsv)"

# shellcheck disable=SC2329  # called by the trap
restore() {
    local status=$?
    trap - EXIT
    if [ "$min_before" != 1 ]; then
        az containerapp update -n "$app" -g "$RG" --min-replicas "${min_before:-0}" \
            --max-replicas 1 --output none
    fi
    exit "$status"
}
trap restore EXIT

az containerapp update -n "$app" -g "$RG" --container-name "$container" --image "$image" \
    --min-replicas 1 --max-replicas 1 --output none
revision="$(az containerapp show -n "$app" -g "$RG" \
    --query properties.latestRevisionName -o tsv)"

for _ in $(seq 60); do  # 10 minutes
    state="$(az containerapp revision show -n "$app" -g "$RG" --revision "$revision" \
        --query 'properties.[provisioningState, runningState, healthState]' -o tsv | xargs)"
    case "$state" in
        Provisioned\ Running*\ Healthy)
            echo "$app healthy on $image"
            exit 0
            ;;
    esac
    sleep 10
done
echo "$app not healthy after 10 minutes: $state" >&2
az containerapp revision show -n "$app" -g "$RG" --revision "$revision" \
    --query 'properties.{details: runningStateDetails, error: provisioningError}' -o json >&2
echo "see the logs from your laptop: az containerapp logs show -n $app -g $RG --tail 50" >&2
exit 1
