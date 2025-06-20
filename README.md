# Azure Functions - Guia Passo a Passo

Este guia mostra como configurar uma Azure Function do zero e fazer pull de uma função existente no Azure.

## 📋 Pré-requisitos

- Conta Azure ativa
- Windows 10/11
- Node.js instalado
- Python 3.9+ instalado

## 🛠️ Instalação das Ferramentas

### 1. Instalar Azure CLI
```bash
# Baixar e instalar do site oficial
https://aka.ms/installazurecliwindows

# Verificar instalação
az --version
```

### 2. Instalar Azure Functions Core Tools
```bash
# Via npm (recomendado)
npm install -g azure-functions-core-tools@4 --unsafe-perm true

# Verificar instalação
func --version
```

## 🔐 Configurar Acesso ao Azure

### 1. Login no Azure CLI
```bash
# Login padrão
az login

# Se tiver problemas com MFA, use device code
az login --use-device-code
```

### 2. Verificar Subscription
```bash
# Listar subscriptions
az account list --output table

# Definir subscription ativa (se necessário)
az account set --subscription "sua-subscription-id"
```

## 📥 Fazer Pull de uma Function Existente

### 1. Criar Diretório Local
```bash
# Criar pasta do projeto
mkdir azure-functions
cd azure-functions
```

### 2. Buscar Configurações da Function App
```bash
# Substituir pelos seus valores
func azure functionapp fetch-app-settings NOME-DA-FUNCTION-APP --resource-group NOME-DO-RESOURCE-GROUP

# Exemplo do nosso projeto:
func azure functionapp fetch-app-settings FAAS-Valuation --resource-group FAAS-Valuation_group
```

### 3. Fazer Pull do Código
```bash
# Pull da function app completa
func azure functionapp fetch NOME-DA-FUNCTION-APP --resource-group NOME-DO-RESOURCE-GROUP

# Exemplo:
func azure functionapp fetch FAAS-Valuation --resource-group FAAS-Valuation_group
```

## 🏗️ Criar Nova Function do Zero

### 1. Inicializar Projeto
```bash
# Criar novo projeto Python
func init . --python

# Ou especificar versão
func init . --worker-runtime python --python-version 3.9
```

### 2. Criar Nova Função
```bash
# Criar função HTTP
func new --name HttpTrigger1 --template "HTTP trigger" --authlevel "anonymous"
```

### 3. Estrutura Criada
```
azure-functions/
├── function_app/
│   ├── __init__.py          # Código da função
│   └── function.json        # Configurações
├── host.json                # Configurações globais
├── local.settings.json      # Configurações locais
└── requirements.txt         # Dependências Python
```

## 📁 Estrutura do Projeto

## ⚙️ Configurar Arquivos Essenciais

### 1. local.settings.json
```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "sua-connection-string",
    "FUNCTIONS_WORKER_RUNTIME": "python"
  }
}
```

### 2. host.json
```json
{
  "version": "2.0",
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": true,
        "excludedTypes": "Request"
      }
    }
  }
}
```

### 3. function.json (dentro da pasta da função)
```json
{
  "scriptFile": "__init__.py",
  "bindings": [
    {
      "authLevel": "anonymous",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": ["get", "post"]
    },
    {
      "type": "http",
      "direction": "out",
      "name": "$return"
    }
  ]
}
```

## 🧪 Testar Localmente

### 1. Executar Function Local
```bash
# Iniciar host local
func start

# Saída esperada:
# Functions:
#   HttpTrigger1: [GET,POST] http://localhost:7071/api/HttpTrigger1
```

### 2. Testar Endpoint
```bash
# Via navegador
http://localhost:7071/api/HttpTrigger1

# Via curl
curl http://localhost:7071/api/HttpTrigger1

# Via PowerShell
Invoke-RestMethod -Uri "http://localhost:7071/api/HttpTrigger1" -Method Get
```

## 🚀 Deploy para Azure

### 1. Criar Resource Group (se não existir)
```bash
az group create --name meu-resource-group --location "Brazil South"
```

### 2. Criar Storage Account
```bash
az storage account create --name meustorage123 --location "Brazil South" --resource-group meu-resource-group --sku Standard_LRS
```

### 3. Criar Function App
```bash
az functionapp create --resource-group meu-resource-group --consumption-plan-location "Brazil South" --runtime python --runtime-version 3.9 --functions-version 4 --name minha-function-app --storage-account meustorage123 --os-type linux
```

### 4. Fazer Deploy
```bash
func azure functionapp publish minha-function-app
```

## 🔄 Sincronizar Configurações

### 1. Download das Configurações do Azure
```bash
# Baixar app settings do Azure para local
func azure functionapp fetch-app-settings minha-function-app --resource-group meu-resource-group
```

### 2. Upload das Configurações Locais
```bash
# Subir configurações locais para Azure (cuidado!)
func azure functionapp publish minha-function-app --publish-local-settings -i
```

## 📊 Verificar Deploy

### 1. Testar Function no Azure
```bash
# URL da function será algo como:
https://minha-function-app.azurewebsites.net/api/HttpTrigger1
```

### 2. Ver Logs no Portal
1. Acesse portal.azure.com
2. Vá para sua Function App
3. Menu lateral → Functions → sua função
4. Clique em "Monitor" para ver execuções

## 🔧 Comandos Úteis

```bash
# Ver todas as functions locais
func list

# Ver configurações atuais
func settings list

# Executar function específica localmente
func run HttpTrigger1

# Ver logs detalhados
func start --verbose

# Limpar cache local
func extensions clear
```

## ❌ Solução de Problemas Comuns

### Erro: "host.json not found"
```bash
# Criar host.json manualmente
echo '{"version": "2.0"}' > host.json
```

### Erro: "AzureWebJobsStorage missing"
```bash
# Adicionar no local.settings.json
"AzureWebJobsStorage": "UseDevelopmentStorage=true"
```

### Erro de Login MFA
```bash
# Usar device code
az login --use-device-code
```

### Function não aparece no portal
- Aguarde alguns minutos após deploy
- Verifique se o deploy foi bem-sucedido
- Confira se está no resource group correto

## 📝 Checklist Final

- [ ] Azure CLI instalado e logado
- [ ] Functions Core Tools instalado
- [ ] Projeto local configurado
- [ ] local.settings.json configurado
- [ ] Teste local funcionando
- [ ] Deploy realizado com sucesso
- [ ] Function acessível no Azure
- [ ] Logs funcionando no portal

---

**🎯 Objetivo:** Ter uma Azure Function funcionando localmente e no Azure em menos de 30 minutos!

## 📊 Monitoramento

### Application Insights
- Logs de execução em tempo real
- Métricas de performance
- Rastreamento de erros
- Análise de dependências

### Acesso aos Logs
1. Portal Azure → Function App → Monitoring → Logs
2. Application Insights → Logs (KQL queries)
3. Fluxo de logs em tempo real

## 🔐 Segurança

- Chaves de acesso gerenciadas via Azure Key Vault (recomendado)
- Application Insights para auditoria
- CORS configurado conforme necessário
- Autenticação anônima (apenas para desenvolvimento)

## 🚀 CI/CD

### Configuração Atual
- Deploy manual via Azure Functions Core Tools
- Sincronização de configurações entre local e Azure
- Monitoramento via Application Insights

### Próximos Passos
- [ ] Implementar GitHub Actions para CI/CD automatizado
- [ ] Configurar ambientes de staging/produção
- [ ] Implementar testes automatizados
- [ ] Configurar autenticação para produção

## 📝 Funcionalidades

### Função Principal
- **Nome**: HttpTrigger1
- **Trigger**: HTTP (GET/POST)
- **Resposta**: JSON com mensagem de sucesso
- **Tratamento de Erros**: Retorna erro 500 em caso de exceção

### Integrações
- Azure OpenAI para processamento de IA
- Azure Storage para persistência
- Application Insights para observabilidade

## 🛠️ Comandos Úteis

```bash
# Verificar versão do Core Tools
func --version

# Listar funções locais
func list

# Ver logs em tempo real (local)
func start --verbose

# Testar função específica
func run HttpTrigger1

# Sincronizar configurações
func azure functionapp fetch-app-settings FAAS-Valuation
```

## 📚 Documentação

- [Azure Functions Documentation](https://learn.microsoft.com/en-us/azure/azure-functions/)
- [Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local)
- [Python Developer Guide](https://learn.microsoft.com/en-us/azure/azure-functions/functions-reference-python)


