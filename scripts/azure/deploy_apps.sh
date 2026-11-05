#!/usr/bin/env bash
# Second stage: build and push the images for this commit, then apply the full Bicep with the
# apps. Needs make azure-publish-data and make azure-secrets first. Rerun it after an infra
# change; it resets session settings (LLM URL, min replicas), so not during a session.
source "$(dirname "$0")/common.sh"

git diff --quiet HEAD || die "uncommitted changes: the image tag is the commit, so commit first"
tag="$(git rev-parse HEAD)"
server="$(output registryLoginServer)"
api_image="$server/rig-assist-api:$tag"
ui_image="$server/rig-assist-ui:$tag"

docker build -t "$api_image" .
docker build -t "$ui_image" ui

# Push from a throwaway Docker config: `az acr login` needs the desktop credential store, which
# can be locked behind a GPG passphrase. The token lasts about 3 hours and is never printed.
docker_config="$(mktemp -d)"
trap 'rm -rf "$docker_config"' EXIT
docker_host="$(docker context inspect --format '{{.Endpoints.docker.Host}}')"
registry_docker() { docker --config "$docker_config" -H "$docker_host" "$@"; }
az acr login -n "$(output registryName)" --expose-token --query accessToken -o tsv \
    --only-show-errors \
    | registry_docker login "$server" -u 00000000-0000-0000-0000-000000000000 --password-stdin
registry_docker push "$api_image"
registry_docker push "$ui_image"

deploy stage2 deployApps=true apiImage="$api_image" uiImage="$ui_image"
echo "apps deployed: https://$(az containerapp show -n "$(output uiAppName)" -g "$RG" \
    --query properties.configuration.ingress.fqdn -o tsv)"
