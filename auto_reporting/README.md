# auto_reporting

Apri il terminale Python in VS Code e clona il progetto:

```bash
git clone https://github.com/furio03/4M.git
```

Per aggiungere, eliminare o modificare file, usa sempre i comandi Git.

## Pubblicazione sicura su GitHub (senza chiavi API)

1. Salva le chiavi solo in variabili d'ambiente locali.
2. Non committare mai file .env o chiavi hardcoded nel codice.
3. Usa .env.example come template condivisibile senza valori reali.

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

### Se la chiave era gia nei commit precedenti

- Revoca o rigenera subito la chiave dal provider Groq.
- Riscrivi la history Git per rimuoverla dai vecchi commit prima di rendere pubblico il repository.