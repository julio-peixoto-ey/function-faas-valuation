# Azure Functions - Guia Passo a Passo

Este guia mostra como configurar uma Azure Function do zero e fazer pull de uma função existente no Azure.

## 📋 Pré-requisitos

- Conta Azure ativa
- Windows 10/11
- Node.js instalado
- Python 3.9+ instalado

## 🛠️ Instalação das Ferramentas

### 1. Instalar Azure Functions Core Tools
```bash
# Via npm (recomendado)
npm install -g azure-functions-core-tools@4 --unsafe-perm true

# Verificar instalação
func --version
```

### 2. Instalar uv (Gerenciador de Pacotes Python)

#### Instalação do uv
```bash
# Via pip
pip install uv

# Via PowerShell (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Via curl (WSL/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verificar instalação
uv --version
```

#### Configurar Projeto com uv
```bash
# Inicializar projeto Python com uv
uv init

# Criar ambiente virtual
uv venv

# Ativar ambiente virtual (Windows)
.venv\Scripts\activate

# Ativar ambiente virtual (WSL/Linux)
source .venv/bin/activate

# Instalar dependências do projeto
uv sync

# Adicionar nova dependência
uv add package-name

# Adicionar dependência de desenvolvimento
uv add --dev package-name
```

#### Comandos Úteis do uv
```bash
# Ver dependências instaladas
uv pip list

# Atualizar todas as dependências
uv sync --upgrade

# Executar comando no ambiente virtual
uv run python script.py

# Executar função local com uv
uv run func start

# Verificar dependências desatualizadas
uv pip list --outdated

# Remover dependência
uv remove package-name

# Exportar requirements.txt
uv pip freeze > requirements.txt
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


