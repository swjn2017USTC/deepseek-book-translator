# Security

Never commit API keys. The runtime reads the DeepSeek key from `DEEPSEEK_API_KEY`; JSON configuration accepts the environment variable name, not the secret value. Rotate a key immediately if it appears in a commit, log, screenshot, issue, or release archive, and remove it from Git history before publishing.

Before every push, inspect `git status --short`, `git diff --cached`, and the staged file list. This repository ignores normal work and output directories, but custom paths remain the user's responsibility. Report suspected vulnerabilities privately to the repository owner rather than opening a public issue containing secrets or private text.
