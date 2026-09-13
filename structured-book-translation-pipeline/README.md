# Structured Book Translation Pipeline

This project cleans PaddleOCR text, translates it through DeepSeek's official API, and renders a structure-bound Markdown book. The sibling `chapter-structure-recovery-lab/` provides the chapter tree. See the repository root README for installation and the new-book workflow.

The default provider uses `DEEPSEEK_API_KEY`, `https://api.deepseek.com/chat/completions`, `deepseek-v4-flash`, and Bearer authentication. No key is stored in project files. `python new_book.py --help` lists the workflow commands.
