import os


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
	raise RuntimeError(
		"Missing GROQ_API_KEY environment variable. "
		"Set it before running the app, e.g. 'export GROQ_API_KEY=...'."
	)
