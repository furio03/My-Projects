# 4M

# open your python terminal on VS CODE and use: 
# git clone https://github.com/furio03/4M.git

# For adding, deleting and modify a file you MUST use Git commands. Ask chat gpt if you do not know how to do it.

## Pubblicazione sicura su GitHub (senza chiavi API)

1. Salva le chiavi solo in variabili d'ambiente locali.
2. Non committare mai file `.env` o chiavi hardcoded nel codice.
3. Usa `.env.example` come template condivisibile (senza valori reali).

### Setup locale chiave Groq

Nel terminale:

```bash
export GROQ_API_KEY="la_tua_chiave_reale"
```

Poi avvia il progetto normalmente.

### Primo push consigliato

```bash
git add .
git commit -m "Secure secrets: move API key to env vars"
git push origin main
```

### Se la chiave era gia' nei commit precedenti

- Revoca/rigenera subito la chiave dal provider (Groq).
- Riscrivi la history Git per rimuoverla dai vecchi commit prima di rendere pubblico il repo.