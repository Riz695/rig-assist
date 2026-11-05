// The three Container Apps: ui (public, one allowed IP) -> api (internal) -> qdrant (internal).
// Each replica starts with an init container that copies its data from Blob into an EmptyDir
// volume, using the managed identity. Nothing written in Azure survives a replica.

param location string
param namePrefix string
param environmentId string
param identityId string
param identityClientId string
param registryServer string
param blobEndpoint string
param dataContainer string
param keyVaultUri string
param allowedIp string
param apiImage string
param uiImage string
param qdrantImage string
param llmBaseUrl string
param llmModel string
param apiMinReplicas int

var qdrantName = '${namePrefix}-qdrant'
var apiName = '${namePrefix}-api'
var uiName = '${namePrefix}-ui'
var collection = 'rig_assist_chunks'
var snapshotDir = '/qdrant/snapshots'

var appIdentity = {
  type: 'UserAssigned'
  userAssignedIdentities: {
    '${identityId}': {}
  }
}
var registries = [
  {
    server: registryServer
    identity: identityId
  }
]

// Runs from the API image, which contains rig_assist.deploy.blob_fetch.
// Consumption counts init containers in the replica total, which must be a valid CPU/memory pair.
var initResources = {
  cpu: json('0.25')
  memory: '0.5Gi'
}
func blobFetchArgs(endpoint string, container string, maps array) array =>
  concat(
    ['-m', 'rig_assist.deploy.blob_fetch', '--account-url', endpoint, '--container', container],
    flatten(map(maps, item => ['--map', item]))
  )

resource qdrant 'Microsoft.App/containerApps@2026-07-01' = {
  name: qdrantName
  location: location
  identity: appIdentity
  properties: {
    environmentId: environmentId
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      registries: registries
      ingress: {
        external: false
        targetPort: 6333
        allowInsecure: true // plain HTTP inside the environment
      }
    }
    template: {
      volumes: [
        {
          name: 'snapshots'
          storageType: 'EmptyDir'
        }
      ]
      initContainers: [
        {
          name: 'fetch-snapshot'
          image: apiImage
          command: ['python']
          args: blobFetchArgs(blobEndpoint, dataContainer, ['qdrant/=${snapshotDir}'])
          env: [
            {
              name: 'AZURE_CLIENT_ID'
              value: identityClientId
            }
          ]
          resources: initResources
          volumeMounts: [
            {
              volumeName: 'snapshots'
              mountPath: snapshotDir
            }
          ]
        }
      ]
      containers: [
        {
          name: 'qdrant'
          image: qdrantImage
          // Container Apps' args replace the image's CMD, which is what starts Qdrant.
          command: ['./entrypoint.sh']
          args: ['--snapshot', '${snapshotDir}/${collection}.snapshot:${collection}']
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          volumeMounts: [
            {
              volumeName: 'snapshots'
              mountPath: snapshotDir
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

resource api 'Microsoft.App/containerApps@2026-07-01' = {
  name: apiName
  location: location
  identity: appIdentity
  properties: {
    environmentId: environmentId
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      registries: registries
      ingress: {
        external: false
        targetPort: 8000
        allowInsecure: true // plain HTTP inside the environment
      }
      secrets: [
        {
          name: 'llm-api-key'
          keyVaultUrl: '${keyVaultUri}secrets/llm-api-key'
          identity: identityId
        }
        {
          name: 'langfuse-public-key'
          keyVaultUrl: '${keyVaultUri}secrets/langfuse-public-key'
          identity: identityId
        }
        {
          name: 'langfuse-secret-key'
          keyVaultUrl: '${keyVaultUri}secrets/langfuse-secret-key'
          identity: identityId
        }
      ]
    }
    template: {
      volumes: [
        {
          name: 'data'
          storageType: 'EmptyDir'
        }
      ]
      initContainers: [
        {
          name: 'fetch-data'
          image: apiImage
          command: ['python']
          args: blobFetchArgs(blobEndpoint, dataContainer, [
            'index/=/app/data/index'
            'raw/=/app/data/raw'
          ])
          env: [
            {
              name: 'AZURE_CLIENT_ID'
              value: identityClientId
            }
          ]
          resources: initResources
          volumeMounts: [
            {
              volumeName: 'data'
              mountPath: '/app/data'
            }
          ]
        }
      ]
      containers: [
        {
          name: 'api'
          image: apiImage
          env: [
            {
              // Port 80 is the ingress; without a port qdrant-client would use 6333.
              name: 'QDRANT_URL'
              value: 'http://${qdrant.name}:80'
            }
            {
              name: 'LLM_PROVIDER'
              value: 'openai_compatible'
            }
            {
              name: 'LLM_BASE_URL'
              value: llmBaseUrl
            }
            {
              name: 'LLM_MODEL'
              value: llmModel
            }
            {
              name: 'LLM_THINKING'
              value: 'off'
            }
            {
              name: 'LLM_API_KEY'
              secretRef: 'llm-api-key'
            }
            {
              name: 'LANGFUSE_PUBLIC_KEY'
              secretRef: 'langfuse-public-key'
            }
            {
              name: 'LANGFUSE_SECRET_KEY'
              secretRef: 'langfuse-secret-key'
            }
          ]
          resources: {
            cpu: json('1')
            memory: '2Gi'
          }
          volumeMounts: [
            {
              volumeName: 'data'
              mountPath: '/app/data'
            }
          ]
          probes: [
            {
              // Up to 2.5 min after the init container, then the other probes take over.
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: 8000
              }
              periodSeconds: 15
              failureThreshold: 10
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              periodSeconds: 30
              timeoutSeconds: 5
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/ready'
                port: 8000
              }
              periodSeconds: 10
              timeoutSeconds: 5
            }
          ]
        }
      ]
      scale: {
        minReplicas: apiMinReplicas
        maxReplicas: 1 // the SQLite job store and in-process jobs allow only one
      }
    }
  }
}

resource ui 'Microsoft.App/containerApps@2026-07-01' = {
  name: uiName
  location: location
  identity: appIdentity
  properties: {
    environmentId: environmentId
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      registries: registries
      ingress: {
        external: true
        targetPort: 8080
        ipSecurityRestrictions: [
          {
            name: 'allow-one-ip'
            action: 'Allow'
            ipAddressRange: '${allowedIp}/32'
          }
        ]
      }
    }
    template: {
      containers: [
        {
          name: 'ui'
          image: uiImage
          env: [
            {
              // The full name: nginx's resolver ignores resolv.conf's search list, so a
              // short app name wouldn't resolve. Host is how ingress finds the app.
              name: 'API_UPSTREAM'
              value: 'http://${api.properties.configuration.ingress.fqdn}'
            }
            {
              name: 'API_HOST'
              value: api.properties.configuration.ingress.fqdn
            }
          ]
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
}

output uiFqdn string = ui.properties.configuration.ingress.fqdn
