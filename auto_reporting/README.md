# auto_reporting

Open the Python terminal in VS Code and clone the project:

```bash
git clone https://github.com/furio03/4M.git
```

To add, delete, or modify files, always use Git commands.

## Secure GitHub Publishing (without API keys)

1. Store keys only in local environment variables.
2. Never commit .env files or hardcoded keys in source code.
3. Use .env.example as a shareable template without real values.

### Local Groq key setup

In the terminal:

```bash
export GROQ_API_KEY="your_real_key"
```

Once your key is set, start the app by running `website.py`:

```bash
python app/website.py
```

### Recommended first push

```bash
git add .
git commit -m "Secure secrets: move API key to env vars"
git push origin main
```

### If the key was already committed in previous commits

- Revoke or regenerate the key immediately from the Groq provider.
- Rewrite Git history to remove it from old commits before making the repository public.