// Key Vault with Azure RBAC for access. Bicep creates no secrets, so no value ever passes
// through a template or deployment history: scripts/azure/set_secrets.py writes them.

param location string
param name string

resource vault 'Microsoft.KeyVault/vaults@2026-05-15' = {
  name: name
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    softDeleteRetentionInDays: 7
  }
}

output name string = vault.name
output uri string = vault.properties.vaultUri
