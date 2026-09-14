# Contributing

欢迎提交修复、文档和合成测试。请先在 issue 中说明较大的功能改动；小修复可直接提交 PR。

1. 从 `main` 创建分支，只提交你有权授权的代码、文本和素材。不要提交真实书籍 OCR、译文、封面、个人路径、API key 或其他隐私数据。
2. 代码更改请在仓库根目录运行两组测试和离线流程检查：

   ```bash
   (cd chapter-structure-recovery-lab && python -m pytest -q)
   (cd structured-book-translation-pipeline && python -m pytest -q)
   python deepseek_book_translator_gui.py --offline-smoke
   ```

3. PR 请说明触发问题的输入、修复后的行为和运行过的检查。涉及 Windows EXE 时，查看 Actions 的 **Build Windows EXE** 测试结果。
4. 安全漏洞请按 [SECURITY.md](SECURITY.md) 私下报告，不要公开含有密钥或书籍内容的 issue。

贡献内容在你有权授权的范围内按仓库的 [MIT 许可证](LICENSE) 发布；第三方依赖仍遵循各自许可证。
