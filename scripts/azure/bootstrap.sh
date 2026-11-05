#!/usr/bin/env bash
# First stage: the supporting resources (registry, storage, vault, identity, environment), your
# rights to fill them, and the GitHub identity deploys sign in with. Safe to rerun.
source "$(dirname "$0")/common.sh"

BRANCHES=(main phase-8-azure)  # the branch credential is deleted once the merge is verified
PROVIDERS=(Microsoft.App Microsoft.ContainerRegistry Microsoft.KeyVault Microsoft.ManagedIdentity
    Microsoft.OperationalInsights Microsoft.Storage)

grant() {
    az role assignment create --assignee-object-id "$1" --assignee-principal-type "$2" \
        --role "$3" --scope "$4" --output none
}

for provider in "${PROVIDERS[@]}"; do  # a new subscription has them unregistered
    az provider register --namespace "$provider" --wait --output none
done
az group create -n "$RG" -l "$LOCATION" --output none
deploy base deployApps=false
echo "supporting resources deployed in $RG ($LOCATION)"

registry_id="$(az acr show -n "$(output registryName)" --query id -o tsv)"
storage_id="$(az storage account show -n "$(output storageAccountName)" -g "$RG" --query id -o tsv)"
vault_id="$(az keyvault show -n "$(output keyVaultName)" --query id -o tsv)"
group_id="$(az group show -n "$RG" --query id -o tsv)"

# Owner of the subscription still has no access to blob or secret contents.
me="$(az ad signed-in-user show --query id -o tsv)"
grant "$me" User "Storage Blob Data Contributor" "$storage_id"
grant "$me" User "Key Vault Secrets Officer" "$vault_id"

app_id="$(az ad app list --display-name "$GITHUB_APP" --query '[0].appId' -o tsv)"
if [ -z "$app_id" ]; then
    app_id="$(az ad app create --display-name "$GITHUB_APP" --query appId -o tsv)"
fi
if ! az ad sp show --id "$app_id" --output none 2>/dev/null; then
    az ad sp create --id "$app_id" --output none
fi
principal_id="$(az ad sp show --id "$app_id" --query id -o tsv)"

repo="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
# GitHub decides the token's subject: a repo can use immutable ids instead of owner/name.
subject_prefix="$(gh api "repos/$repo/actions/oidc/customization/sub" -q .sub_claim_prefix)"
for branch in "${BRANCHES[@]}"; do
    parameters="{\"name\": \"github-$branch\", \"issuer\": \"https://token.actions.githubusercontent.com\", \"subject\": \"$subject_prefix:ref:refs/heads/$branch\", \"audiences\": [\"api://AzureADTokenExchange\"]}"
    existing="$(az ad app federated-credential list --id "$app_id" \
        --query "[?name=='github-$branch'] | length(@)" -o tsv)"
    if [ "$existing" = 0 ]; then
        az ad app federated-credential create --id "$app_id" --output none --parameters "$parameters"
    else
        az ad app federated-credential update --id "$app_id" --output none \
            --federated-credential-id "github-$branch" --parameters "$parameters"
    fi
done

# Only what a tag swap needs; applying the Bicep stays on the laptop.
grant "$principal_id" ServicePrincipal AcrPush "$registry_id"
grant "$principal_id" ServicePrincipal "Container Apps Contributor" "$group_id"

gh variable set AZURE_CLIENT_ID --body "$app_id"
gh variable set AZURE_TENANT_ID --body "$(az account show --query tenantId -o tsv)"
gh variable set AZURE_SUBSCRIPTION_ID --body "$(az account show --query id -o tsv)"
gh variable set AZURE_RESOURCE_GROUP --body "$RG"
gh variable set ACR_NAME --body "$(output registryName)"
gh variable set API_APP --body "$(output apiAppName)"
gh variable set UI_APP --body "$(output uiAppName)"
echo "next: make azure-publish-data, make azure-secrets, make azure-apps"
