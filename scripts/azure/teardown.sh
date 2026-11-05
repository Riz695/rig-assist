#!/usr/bin/env bash
# Delete everything, permanently, then check it's gone: the resource group, the Key Vault's
# soft-deleted copy (it would hold the name for 7 days) and the GitHub identity.
source "$(dirname "$0")/common.sh"

read -r -p "This permanently deletes resource group $RG and its data. Type its name to go on: " answer
[ "$answer" = "$RG" ] || die "nothing deleted"

if [ "$(az group exists -n "$RG")" = true ]; then
    az group delete -n "$RG" --yes
fi
vaults="$(az keyvault list-deleted --resource-type vault \
    --query "[?contains(properties.vaultId, '/$RG/')].name" -o tsv)"
for vault in $vaults; do
    az keyvault purge --name "$vault"
done
for app_id in $(az ad app list --display-name "$GITHUB_APP" --query '[].appId' -o tsv); do
    az ad app delete --id "$app_id"
done
gh variable set AZURE_DEPLOY_ENABLED --body false

failed=0
check() {
    if [ "$2" = "$3" ]; then
        echo "ok: $1"
    else
        echo "NOT DONE: $1 (found $2)"
        failed=1
    fi
}
check "resource group $RG gone" "$(az group exists -n "$RG")" false
check "no soft-deleted Key Vault in the subscription" \
    "$(az keyvault list-deleted --resource-type vault --query 'length(@)' -o tsv)" 0
check "app registration $GITHUB_APP gone" \
    "$(az ad app list --display-name "$GITHUB_APP" --query 'length(@)' -o tsv)" 0
check "deploys switched off" \
    "$(gh variable list --json name,value -q '.[] | select(.name == "AZURE_DEPLOY_ENABLED") | .value')" false
exit "$failed"
