// Log Analytics workspace: where the Container Apps environment sends container logs.

param location string
param name string

resource workspace 'Microsoft.OperationalInsights/workspaces@2026-03-01' = {
  name: name
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

output name string = workspace.name
