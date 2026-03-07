# Deploying bakup.ai for Teams

> **Goal:** Run bakup.ai on a single server so 100+ developers and testers
> can access it from their browsers—no local install required.

---

## Architecture at a Glance

```
┌─────────┐        ┌──────────────────────────────────────┐
│ Browser  │──HTTP──▶  nginx (:80)                         │
│ (Users)  │        │   ├─ /            → static UI files  │
│          │        │   ├─ /ask         → backend :8000    │
│          │        │   ├─ /ask/stream  → backend (SSE)    │
│          │        │   ├─ /index       → backend          │
│          │        │   └─ /health …    → backend          │
│          │        ├──────────────────────────────────────│
│          │        │  backend (:8000)                      │
│          │        │   ├─ FastAPI + uvicorn                │
│          │        │   ├─ sentence-transformers (embed)    │
│          │        │   ├─ ChromaDB (vector store)          │
│          │        │   └─ LLM (Ollama / OpenAI / Azure)   │
└─────────┘        └──────────────────────────────────────┘
```

All processing happens server-side. Users only need a modern browser.

---

## Prerequisites

| What | Minimum | Recommended |
|------|---------|-------------|
| **CPU** | 2 cores | 4 cores |
| **RAM** | 4 GB | 8 GB |
| **Disk** | 10 GB | 20 GB SSD |
| **Docker** | v24+ | latest |
| **Docker Compose** | v2+ | latest |
| **OS** | Any Linux (Ubuntu 22.04 recommended) | — |

---

## Option A: Deploy on a VPS (DigitalOcean / Hetzner / Linode)

This is the **simplest and cheapest** option for 100+ users.

### 1. Provision a server

| Provider | Plan | Cost |
|----------|------|------|
| DigitalOcean | 4 GB / 2 vCPU Droplet | ~$24/mo |
| Hetzner | CX31 (8 GB / 2 vCPU) | ~€7/mo |
| Linode | 4 GB Shared | ~$24/mo |
| AWS EC2 | t3.medium (4 GB) | ~$30/mo |

### 2. Install Docker

```bash
# Ubuntu / Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in, then:
docker --version
docker compose version
```

### 3. Clone and configure

```bash
git clone https://github.com/YOUR_ORG/bakup.ai.git
cd bakup.ai

# Create env file
cp .env.example .env

# Set a strong access key (shared with your team)
nano .env
# → BAKUP_ACCESS_KEY=your-secret-key-here
```

### 4. Configure LLM provider

bakup.ai supports three LLM backends. For **multi-user production**, use
OpenAI or Azure—they handle concurrency much better than local Ollama.

**Option 1 — OpenAI (recommended for teams):**
```env
BAKUP_LLM_PROVIDER=openai
BAKUP_OPENAI_API_KEY=sk-...
BAKUP_OPENAI_MODEL=gpt-4o-mini
```

**Option 2 — Azure OpenAI:**
```env
BAKUP_LLM_PROVIDER=azure
BAKUP_AZURE_ENDPOINT=https://YOUR.openai.azure.com
BAKUP_AZURE_API_KEY=...
BAKUP_AZURE_DEPLOYMENT=gpt-4o-mini
BAKUP_AZURE_API_VERSION=2024-02-15-preview
```

**Option 3 — Ollama (self-hosted, single-user or small teams):**
```bash
# Install Ollama on the server
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3
```
```env
BAKUP_LLM_PROVIDER=ollama
BAKUP_OLLAMA_BASE_URL=http://host.docker.internal:11434
BAKUP_LLM_MODEL=llama3
```

### 5. Open port to the network

The docker-compose.yml already binds to `0.0.0.0:8080`. For production on
port 80 or 443, edit the UI service ports:

```yaml
# docker-compose.yml → services → ui → ports
ports:
  - "0.0.0.0:80:80"
```

### 6. Launch

```bash
# Create required host directories
mkdir -p projects logs

# Start in detached mode
docker compose up -d

# Watch logs until healthy
docker compose logs -f
```

First boot takes 2–5 minutes (downloads the 80 MB embedding model).

### 7. Verify

```bash
# Health check
curl http://localhost:8080/health

# From your laptop (replace SERVER_IP)
curl http://SERVER_IP:8080/health
```

Share `http://SERVER_IP:8080` with your team. They enter the access key
and start using bakup.ai immediately.

---

## Option B: Deploy on Railway

[Railway](https://railway.app) is a managed PaaS that runs Docker containers
with persistent volumes.

1. Push code to GitHub
2. Create a new Railway project → "Deploy from GitHub repo"
3. Railway auto-detects `docker-compose.yml`
4. Add environment variables in the Railway dashboard
5. Attach a persistent volume at `/app/vectordb`
6. Deploy

**Cost:** ~$5–20/mo depending on usage.

---

## Option C: Deploy on Render

[Render](https://render.com) supports Docker with persistent disks.

1. Create a new **Web Service** → Docker
2. Point to your GitHub repo, set root directory to `backend/`
3. Add env vars in the Render dashboard
4. Attach a **Disk** mounted at `/app/vectordb`
5. Create a separate **Static Site** for the `ui/` folder
6. Deploy

**Cost:** ~$7–25/mo.

---

## Adding HTTPS (SSL)

### Option 1: Cloudflare Tunnel (easiest, free)

```bash
# Install cloudflared on the server
curl -fsSL https://pkg.cloudflare.com/cloudflared-stable-linux-amd64.deb -o cf.deb
sudo dpkg -i cf.deb

# Authenticate and create tunnel
cloudflared tunnel login
cloudflared tunnel create bakup
cloudflared tunnel route dns bakup bakup.yourdomain.com

# Run the tunnel (proxies HTTPS → localhost:8080)
cloudflared tunnel --url http://localhost:8080 run bakup
```

### Option 2: Let's Encrypt with Caddy

Replace the nginx UI container with Caddy for automatic HTTPS:

```yaml
# docker-compose.override.yml
services:
  ui:
    image: caddy:2-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - ./ui:/srv:ro
      - caddy-data:/data
volumes:
  caddy-data:
```

```
# Caddyfile
bakup.yourdomain.com {
    root * /srv
    file_server

    handle /health  { reverse_proxy backend:8000 }
    handle /ask/*   { reverse_proxy backend:8000 }
    handle /ask     { reverse_proxy backend:8000 }
    handle /index   { reverse_proxy backend:8000 }
    handle /llm/*   { reverse_proxy backend:8000 }
    handle /debug/* { reverse_proxy backend:8000 }
    handle /download/* { reverse_proxy backend:8000 }
}
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BAKUP_ACCESS_KEY` | *(required)* | Shared key for UI authentication |
| `BAKUP_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` in Docker) |
| `BAKUP_PORT` | `8000` | Backend port |
| `BAKUP_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `BAKUP_CHROMA_DIR` | `./vectordb` | ChromaDB storage path |
| `BAKUP_CHROMA_COLLECTION` | `bakup_default` | Collection name |
| `BAKUP_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `BAKUP_LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `azure` |
| `BAKUP_CORS_ORIGINS` | *(empty)* | Extra allowed origins (comma-separated) |
| `BAKUP_RETRIEVAL_TOP_K` | `8` | Number of chunks to retrieve |
| `BAKUP_CONFIDENCE_THRESHOLD` | `0.35` | Minimum similarity score |

See [.env.example](.env.example) for the full list.

---

## Scaling for More Users

### 100–500 concurrent users

The default single-container setup handles this fine because:
- **Embedding** runs once per index operation, not per query
- **ChromaDB queries** are sub-100ms
- **LLM calls** (OpenAI/Azure) are async and non-blocking
- **nginx** handles thousands of concurrent connections

### 500+ concurrent users

1. Run multiple backend replicas behind a load balancer:
   ```yaml
   services:
     backend:
       deploy:
         replicas: 3
   ```
2. Use a shared ChromaDB volume or switch to ChromaDB client-server mode
3. Consider a dedicated GPU server if using local LLM

### Performance tips

- Use **OpenAI/Azure** instead of Ollama for concurrent LLM calls
- The embedding model loads once into memory (~500 MB) and stays resident
- ChromaDB uses memory-mapped files — more RAM = faster queries
- Set `BAKUP_LOG_LEVEL=WARNING` in production to reduce I/O

---

## Monitoring

```bash
# Live logs
docker compose logs -f

# Health check
curl http://localhost:8080/health

# Resource usage
docker stats

# Restart after config change
docker compose restart backend
```

---

## Updating

```bash
cd bakup.ai
git pull
docker compose build
docker compose up -d
```

Your vector database and model weights persist across updates (stored in
named Docker volumes).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Backend won't start | Check `.env` has `BAKUP_ACCESS_KEY` set |
| "Model not found" | Ensure LLM provider env vars are set |
| Slow first query | Normal — embedding model loads on first use (~30s) |
| Can't access from network | Ensure port binding is `0.0.0.0:8080:80` not `127.0.0.1` |
| "Connection refused" on /ask | Wait for healthcheck to pass (first boot: ~90s) |
| Out of memory | Increase server RAM to 8 GB or use lighter embedding model |
