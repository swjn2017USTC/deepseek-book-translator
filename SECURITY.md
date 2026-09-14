# Security

Never commit API keys. The runtime reads the DeepSeek key from `DEEPSEEK_API_KEY`; JSON configuration accepts the environment variable name, not the secret value. Rotate a key immediately if it appears in a commit, log, screenshot, issue, or release archive, and remove it from Git history before publishing.

Before every push, inspect `git status --short`, `git diff --cached`, and the staged file list. This repository ignores normal work and output directories, but custom paths remain the user's responsibility.

## Private vulnerability reports

Email [swjn2017ustc@gmail.com](mailto:swjn2017ustc@gmail.com) with a concise description, affected version, reproduction steps, and impact. Use a minimal synthetic example; never send a live API key, full OCR book, private translation, or other sensitive data. Please allow time for a private fix before public disclosure. Do not open a public issue with exploit details or secrets.

If GitHub private vulnerability reporting is enabled for this repository, you can also use [Report a vulnerability](https://github.com/swjn2017USTC/deepseek-book-translator/security/advisories/new). The email address above remains the working private channel regardless of that setting.
