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

az acr login -n "$(output registryName)" --only-show-errors
docker build -t "$api_image" .
docker build -t "$ui_image" ui
docker push "$api_image"
docker push "$ui_image"

deploy apps deployApps=true apiImage="$api_image" uiImage="$ui_image"
echo "apps deployed: https://$(az containerapp show -n "$(output uiAppName)" -g "$RG" \
    --query properties.configuration.ingress.fqdn -o tsv)"
