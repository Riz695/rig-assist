// Rig Assist on Azure Container Apps. Scope: one resource group; deleting it removes everything
// here. Applied from the laptop only (scripts/azure); CI/CD only swaps image tags. See
// infra/README.md for the parameters and what each module creates.

targetScope = 'resourceGroup'

@description('Azure region for every resource.')
param location string = resourceGroup().location

@description('Start of every resource name.')
@minLength(3)
@maxLength(12)
param namePrefix string = 'rigassist'

@description('The only IP address allowed to reach the UI. Passed at deploy time, never committed.')
param allowedIp string

@description('False creates only the supporting resources: images, data and secrets must exist before the apps can start.')
param deployApps bool = true

@description('API image, including the registry.')
param apiImage string = ''

@description('UI image, including the registry.')
param uiImage string = ''

@description('Same pinned image as docker-compose.yml.')
param qdrantImage string = 'docker.io/qdrant/qdrant:v1.19.2-unprivileged@sha256:efb96a9425a90d2d5a1a0a474156280df1892dcdf1af3e8515bd2589b1bfd88b'

@description('OpenAI-compatible LLM URL. Settings refuses an empty one, so between sessions this is a name that can never resolve.')
param llmBaseUrl string = 'http://llm-not-started.invalid/v1'

@description('Model the LLM server serves.')
param llmModel string = 'Qwen/Qwen3.6-35B-A3B-FP8'

@description('0 scales the API to zero when idle; 1 keeps it warm during a session.')
@minValue(0)
@maxValue(1)
param apiMinReplicas int = 0

// Globally unique names, stable across redeploys of the same resource group.
var suffix = uniqueString(resourceGroup().id)
var dataContainer = 'data'

module monitoring 'modules/monitoring.bicep' = {
  name: 'monitoring'
  params: {
    location: location
    name: '${namePrefix}-logs'
  }
}

module registry 'modules/registry.bicep' = {
  name: 'registry'
  params: {
    location: location
    name: '${namePrefix}${suffix}'
  }
}

module storage 'modules/storage.bicep' = {
  name: 'storage'
  params: {
    location: location
    name: take('${namePrefix}${suffix}', 24)
    containerName: dataContainer
  }
}

module keyvault 'modules/keyvault.bicep' = {
  name: 'keyvault'
  params: {
    location: location
    name: take('${namePrefix}-${suffix}', 24)
  }
}

module identity 'modules/identity.bicep' = {
  name: 'identity'
  params: {
    location: location
    name: '${namePrefix}-apps'
    registryName: registry.outputs.name
    storageAccountName: storage.outputs.name
    keyVaultName: keyvault.outputs.name
  }
}

module environment 'modules/environment.bicep' = {
  name: 'environment'
  params: {
    location: location
    name: '${namePrefix}-env'
    workspaceName: monitoring.outputs.name
  }
}

module apps 'modules/apps.bicep' = if (deployApps) {
  name: 'apps'
  params: {
    location: location
    namePrefix: namePrefix
    environmentId: environment.outputs.id
    identityId: identity.outputs.id
    identityClientId: identity.outputs.clientId
    registryServer: registry.outputs.loginServer
    blobEndpoint: storage.outputs.blobEndpoint
    dataContainer: dataContainer
    keyVaultUri: keyvault.outputs.uri
    allowedIp: allowedIp
    apiImage: apiImage
    uiImage: uiImage
    qdrantImage: qdrantImage
    llmBaseUrl: llmBaseUrl
    llmModel: llmModel
    apiMinReplicas: apiMinReplicas
  }
}

output uiUrl string = deployApps ? 'https://${apps!.outputs.uiFqdn}' : ''
output registryLoginServer string = registry.outputs.loginServer
output registryName string = registry.outputs.name
output storageAccountName string = storage.outputs.name
output keyVaultName string = keyvault.outputs.name
output apiAppName string = '${namePrefix}-api'
output uiAppName string = '${namePrefix}-ui'
output identityClientId string = identity.outputs.clientId
