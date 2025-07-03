# 🚀 NEW ALGORITHM RAG

## 📖 Visão Geral

![alt text](image.png)

O **NEW ALGORITHM RAG** é um sistema inovador de Retrieval-Augmented Generation (RAG) que revoluciona a forma como informações são processadas e recuperadas em bancos de dados vetoriais. Este algoritmo utiliza uma abordagem multi-dimensional para criar embeddings semânticos mais eficazes e precisos.

### 🎯 Objetivo

Para bancos de dados relacionais com capacidade de pesquisa vetorial, este algoritmo permite que a **informação A** seja relacionada à **informação B** quando existe algum tipo de relação de significado entre elas, expandindo significativamente a capacidade de recuperação de informações relevantes.

## 🔬 Metodologia Inovadora

O algoritmo processa cada chunk de texto através de **três dimensões semânticas distintas**:

### 1. 🎭 Similaridade Semântica
**Objetivo:** Encontrar itens com significado muito parecido
- Gera variações do texto original mantendo o mesmo significado
- Utiliza sinônimos e reestruturação de frases
- Preserva a semântica original integralmente

### 2. 🔗 Relacionamento Semântico
**Objetivo:** Encontrar itens que não significam o mesmo, mas estão relacionados conceitualmente
- Cria textos conceitualmente ligados ao tema central
- Expande o universo semântico sem repetir o conteúdo original
- Estabelece conexões temáticas mais amplas

### 3. 🌐 Contexto Compartilhado (Coocorrência)
**Objetivo:** Encontrar itens que costumam aparecer juntos em situações ou textos
- Gera textos que compartilham o mesmo cenário ou ambiente
- Representa situações comuns no mesmo universo temático
- Captura relações contextuais implícitas

## 🏗️ Arquitetura do Sistema

### Estrutura de Diretórios
```
src/
├── main.py                     # Ponto de entrada da aplicação
├── controller/api/
│   └── router.py              # Rotas da API REST
├── infrastructure/
│   ├── connection_openai.py   # Conexão com OpenAI API
│   └── connection_postgresql.py # Conexão com PostgreSQL
├── models/
│   └── database_models.py     # Modelos SQLAlchemy
├── prompt/
│   ├── similaridade_semantica.txt
│   ├── relacionamento_semantico.txt
│   └── contexto_compartilhado.txt
├── service/
│   └── embedding_service.py   # Serviços de embedding
└── usecase/
    └── embedding_usecase.py   # Casos de uso principais
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.12+**
- **FastAPI** - Framework web moderno e rápido
- **Azure PostgreSQL** com extensão **pgvector** - Banco de dados vetorial
- **OpenAI API** - Geração de texto e embeddings (text-embedding-3-large)
- **LangChain** - Processamento e splitting de texto
- **SQLAlchemy** - ORM para Python

## 📊 Benefícios do Algoritmo

### ✅ Ampliação da Recuperação
- **Maior cobertura de palavras-chave** entre uma pergunta e possíveis textos relevantes
- **Redução do risco de "missed answers"** (respostas perdidas por limitação lexical)
- **Melhoria na precisão** das respostas recuperadas

### ✅ Recuperação Multi-nível
- **Diferentes níveis de granularidade** na busca por informações
- **Flexibilidade** para encontrar informações em contextos variados
- **Robustez** contra variações linguísticas e estilísticas

### ✅ Interpretação de Distâncias Vetoriais
```
0.0 - 0.3   = Muito similar (quase idêntico)
0.3 - 0.6   = Similar (mesmo tópico/contexto)  
0.6 - 0.9   = Relacionado (conceitos próximos)
0.9 - 1.2   = Pouco relacionado
1.2+        = Não relacionado
```

## 🔧 Instalação e Configuração

### Pré-requisitos
- Python 3.12+
- PostgreSQL com extensão pgvector (Azure Database for PostgreSQL recomendado)
- Conta OpenAI/Azure OpenAI com acesso à API

### 1. Preparação do Ambiente

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y
sudo apt install python3.12 python3.12-venv python3.12-dev git -y

# macOS (com Homebrew)
brew update
brew install python@3.12 git
```

### 2. Configuração do PostgreSQL no Azure

#### **🌐 Configuração via Azure Portal:**

1. **Acessar o Portal Azure**
    - Faça login em https://portal.azure.com
    - Na barra de pesquisa, digite "Azure Database for PostgreSQL"
    - Selecione "Azure Database for PostgreSQL flexible servers"

2. **Criar Novo Servidor**
    - Clique em "Create" ou "+ Add"
    - Preencha as informações básicas:
      - **Subscription**: Sua assinatura Azure
      - **Resource Group**: Crie um novo ou use existente
      - **Server name**: `rag-postgres-server` (deve ser único globalmente)
      - **Region**: Escolha a região mais próxima
      - **PostgreSQL version**: 16 (recomendado)
      - **Workload type**: Development

3. **Configurar Rede**
    - **Connectivity method**: Public access (selected IP addresses)
    - **Firewall rules**: 
      - ✅ Allow public access from any Azure service
      - ✅ Add current client IP address
      - Adicione outros IPs conforme necessário

4. **Ativar Extensão Vector**
    - No Portal Azure, vá até seu servidor PostgreSQL
    - No menu lateral, clique em "Server parameters"
    - Procure pelo parâmetro `azure.extensions`
    - Busque e clique em `VECTOR`
    - Clique em "Save" (o servidor será reiniciado automaticamente)

#### **🖥️ Configuração via pgAdmin 4:**

```bash
# Instalar pgAdmin 4
# Ubuntu/Debian
sudo apt install pgadmin4

# macOS
brew install --cask pgadmin4
```

**Configurar Conexão Azure:**
- **Host**: `<servidor>.postgres.database.azure.com`
- **Port**: `5432`
- **Username**: `<seu-usuario>`
- **Password**: `<sua-senha>`
- **SSL Mode**: `require`

### 3. Clone e Setup do Projeto

```bash
# Clonar repositório
git clone https://github.com/Leonardojdss/RAG-new-algorithm.git
cd RAG-new-algorithm

# Criar ambiente virtual
python3.12 -m venv env

# Ativar ambiente virtual
source env/bin/activate  # Linux/macOS
# env\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt
```

### 4. Configuração de Variáveis de Ambiente

```bash
# Criar arquivo .env
touch .env
```

Conteúdo do arquivo `.env`:
```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@server.postgres.database.azure.com:5432/rag_database
POSTGRES_HOST=server.postgres.database.azure.com
POSTGRES_PORT=5432
POSTGRES_DB=rag_database
POSTGRES_USER=username
POSTGRES_PASSWORD=password

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
```

### 5. Criação das Tabelas

Execute no pgAdmin4 ou via psql:

```sql
-- Conectar ao banco
CREATE DATABASE rag_database;

-- Ativar extensão vector
CREATE EXTENSION IF NOT EXISTS vector;

-- Criar tabelas
CREATE TABLE db_origin_text (
    id SERIAL PRIMARY KEY,
    data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE db_correlation_embedding (
    id SERIAL PRIMARY KEY,
    id_text_origin INTEGER NOT NULL REFERENCES db_origin_text(id),
    correlation_type VARCHAR(50) CHECK (correlation_type IN 
        ('Similaridade semântica', 'Relacionamento Semântico', 'Contexto Compartilhado')),
    text_content TEXT NOT NULL,
    vector VECTOR(3072),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Criar índices para performance
CREATE INDEX idx_correlation_vector ON db_correlation_embedding 
USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_correlation_type ON db_correlation_embedding(correlation_type);
CREATE INDEX idx_text_origin ON db_correlation_embedding(id_text_origin);
CREATE INDEX idx_created_at ON db_correlation_embedding(created_at);
```

### 6. Executar a Aplicação

```bash
# Executar aplicação
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## 🚀 Como Usar

O sistema oferece duas funcionalidades principais através de uma API REST. Certifique-se de que a aplicação esteja rodando em `http://localhost:8000` antes de executar os comandos.

### 📝 Gerar Embeddings

Para processar um texto e criar os embeddings nas três dimensões semânticas:

```bash
curl -X 'POST' \
    'http://localhost:8000/new_rag/embedding' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "text": "Inteligência artificial é uma tecnologia que permite às máquinas simular o comportamento humano",
        "index": 5
    }'
```

**Parâmetros:**
- `text`: Texto a ser processado (string em linha única)
- `index`: vai gerar 5 textos de similaridade_semantica, relacionamento_semantico e contexto_compartilhado

### 🔍 Busca Vetorial

Para realizar pesquisas semânticas no banco de dados:

```bash
curl -X 'GET' \
    'http://localhost:8000/new_rag/search_vetorial?question=o%20que%20%C3%A9%20IA&top_k=5' \
    -H 'accept: application/json'
```

**Parâmetros:**
- `question`: Pergunta ou termo de busca
- `top_k`: Número de resultados mais relevantes a retornar (default: 5)

### 🌐 Swagger UI

Acesse a documentação interativa da API em:
```
http://localhost:8000/docs
```

## 🤝 Contribuição

Este projeto representa uma inovação na área de RAG systems. Contribuições são bem-vindas para:
- Otimização de performance
- Novos tipos de correlação semântica
- Melhorias na interface de usuário
- Expansão para outras LLMs

## 📄 Licença

[Defina aqui a licença do projeto]

---

**Desenvolvido com ❤️ para revolucionar a recuperação de informações em sistemas RAG**

