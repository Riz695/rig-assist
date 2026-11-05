// Container registry. No admin user: pushes use Entra sign-in, pulls the apps' managed identity.

param location string
param name string

resource registry 'Microsoft.ContainerRegistry/registries@2025-11-01' = {
  name: name
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

output name string = registry.name
output loginServer string = registry.properties.loginServer
