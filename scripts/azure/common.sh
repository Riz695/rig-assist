# shellcheck shell=bash
# Shared by the scripts here. Names come from the Bicep outputs, so none is written twice.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RG="${RG:-rig-assist}"
LOCATION="${LOCATION:-southeastasia}"
# shellcheck disable=SC2034  # used by the scripts that source this
GITHUB_APP="$RG-github"
# Between sessions: a name that can never resolve (same as main.bicep's default).
# shellcheck disable=SC2034
NO_LLM="http://llm-not-started.invalid/v1"

die() {
    echo "error: $*" >&2
    exit 1
}

# One Bicep output, from the supporting-resources deployment (a failed apps deployment can't lose it).
output() {
    local value
    value="$(az deployment group show -g "$RG" -n base --query "properties.outputs.$1.value" -o tsv)"
    [ -n "$value" ] || die "no Bicep output $1: run make azure-bootstrap first"
    echo "$value"
}

# Looked up at run time and never written to disk or printed: it is personal.
public_ip() {
    local ip url
    for url in https://checkip.amazonaws.com https://icanhazip.com https://api.ipify.org; do
        ip="$(curl -4 -fsS --max-time 10 "$url" 2>/dev/null | tr -d '[:space:]' || true)"
        if [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "$ip"
            return
        fi
    done
    die "could not read this machine's public IPv4"
}

# deploy NAME [PARAM=VALUE ...]: apply the full Bicep. Its output would include the IP, so it's off.
deploy() {
    local name="$1" ip
    shift
    ip="$(public_ip)"
    az deployment group create -g "$RG" -n "$name" --template-file infra/main.bicep \
        --parameters allowedIp="$ip" "$@" --output none
}
