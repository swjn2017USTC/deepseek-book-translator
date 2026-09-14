from __future__ import annotations

"""Small Tkinter front end for the public new-book workflow."""

import json
import os
from pathlib import Path
import platform
import queue
import subprocess
import threading
from typing import Any, Callable, Dict

from .review_gui import ReviewWindow

from .workflow import (
    export_project,
    generate_project_cover,
    initialize_project,
    preflight_project,
    prepare_project,
    project_status,
    render_project,
    translate_project,
)


def validate_book_id(value: str) -> str:
    value = value.strip()
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in value):
        raise ValueError("Book ID 只能包含小写字母、数字、连字符和下划线")
    return value


def self_test() -> Dict[str, Any]:
    import PIL
    import chapter_recovery

    result: Dict[str, Any] = {
        "status": "ok",
        "python": platform.python_version(),
        "pillow": PIL.__version__,
        "chapter_recovery": bool(chapter_recovery),
        "gui_constructed": None,
    }
    if os.name == "nt":
        # The Windows CI build runs this through the frozen executable. This
        # catches missing Tcl/Tk files and widget-construction failures.
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        TranslatorGUI(root)
        root.update_idletasks()
        root.destroy()
        result["gui_constructed"] = True
    return result


class TranslatorGUI:
    def __init__(self, root: Any):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("DeepSeek Book Translator")
        self.root.geometry("980x760")
        self.root.minsize(820, 650)
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.busy = False

        default_project = Path.home() / "Documents" / "DeepSeekBookTranslator" / "my-book"
        self.values: Dict[str, Any] = {
            "ocr": tk.StringVar(),
            "project": tk.StringVar(value=str(default_project)),
            "book_id": tk.StringVar(value="my-book"),
            "title": tk.StringVar(),
            "title_zh": tk.StringVar(),
            "author": tk.StringVar(),
            "source_lang": tk.StringVar(value="英语"),
            "target_lang": tk.StringVar(value="简体中文"),
            "domain": tk.StringVar(value="学术人文社科"),
            "model": tk.StringVar(value=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")),
            "api_key": tk.StringVar(),
            "limit": tk.StringVar(value="10"),
            "theme": tk.StringVar(value="auto"),
        }

        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)

        self._path_row(outer, 0, "OCR JSON", "ocr", self._browse_ocr)
        self._path_row(outer, 1, "项目目录", "project", self._browse_project)
        self._entry_row(outer, 2, "Book ID", "book_id")
        self._entry_row(outer, 3, "原文书名", "title")
        self._entry_row(outer, 4, "中文书名", "title_zh")
        self._entry_row(outer, 5, "作者", "author")
        self._entry_row(outer, 6, "领域", "domain")

        language = ttk.Frame(outer)
        language.grid(row=7, column=1, sticky="ew", pady=4)
        language.columnconfigure((0, 1), weight=1)
        ttk.Label(outer, text="语言").grid(row=7, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(language, textvariable=self.values["source_lang"]).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Entry(language, textvariable=self.values["target_lang"]).grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self._entry_row(outer, 8, "DeepSeek 模型", "model")
        ttk.Label(outer, text="API Key").grid(row=9, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(outer, textvariable=self.values["api_key"], show="●").grid(row=9, column=1, sticky="ew", pady=4)
        ttk.Label(outer, text="仅保存在本程序内存中，不写入项目文件", foreground="#555").grid(row=10, column=1, sticky="w")

        options = ttk.Frame(outer)
        options.grid(row=11, column=1, sticky="ew", pady=(6, 4))
        ttk.Label(outer, text="运行参数").grid(row=11, column=0, sticky="w", padx=(0, 10))
        ttk.Label(options, text="累计翻译数").pack(side="left")
        ttk.Entry(options, textvariable=self.values["limit"], width=8).pack(side="left", padx=(6, 20))
        ttk.Label(options, text="封面主题").pack(side="left")
        ttk.Combobox(
            options, textvariable=self.values["theme"], width=10, state="readonly",
            values=("auto", "ink", "ocean", "ember", "forest", "plum"),
        ).pack(side="left", padx=6)

        buttons = ttk.Frame(outer)
        buttons.grid(row=12, column=0, columnspan=2, sticky="ew", pady=10)
        actions = (
            ("1 初始化项目", self.create_project),
            ("2 准备/检查", lambda: self._run("准备项目", lambda: prepare_project(self._project()))),
            ("3 章节审核", lambda: self.open_review("structure")),
            ("4 术语审核", lambda: self.open_review("glossary")),
            ("5 生成封面", lambda: self._run("生成封面", lambda: generate_project_cover(self._project(), self.values["theme"].get()))),
            ("6 零网络预检", self.preflight),
            ("7 翻译", self.translate),
            ("状态", lambda: self._run("读取状态", lambda: project_status(self._project()))),
            ("渲染 Markdown", lambda: self._run("渲染", lambda: render_project(self._project()))),
            ("导出 PDF+EPUB", lambda: self._run("导出", lambda: export_project(self._project(), "both"))),
            ("打开项目目录", self.open_project),
        )
        for index, (label, action) in enumerate(actions):
            ttk.Button(buttons, text=label, command=action).grid(
                row=index // 5, column=index % 5, sticky="ew", padx=3, pady=3
            )
        for column in range(5):
            buttons.columnconfigure(column, weight=1)

        ttk.Label(outer, text="运行日志").grid(row=13, column=0, columnspan=2, sticky="w")
        self.log = tk.Text(outer, height=15, wrap="word", state="disabled")
        self.log.grid(row=14, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        scrollbar = ttk.Scrollbar(outer, command=self.log.yview)
        scrollbar.grid(row=14, column=2, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)
        outer.rowconfigure(14, weight=1)
        self.root.after(100, self._drain_events)
        self._write("填写书籍信息后先初始化，再准备。遇到结构或术语门禁，请打开对应审核窗口逐项裁决。\n")

    def open_review(self, kind: str) -> None:
        from tkinter import messagebox

        try:
            ReviewWindow(self.root, self._project(), kind, self._run)
        except (FileNotFoundError, ValueError, OSError) as exc:
            messagebox.showerror("无法打开审核", str(exc))

    def _entry_row(self, parent: Any, row: int, label: str, key: str) -> None:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        self.ttk.Entry(parent, textvariable=self.values[key]).grid(row=row, column=1, sticky="ew", pady=4)

    def _path_row(self, parent: Any, row: int, label: str, key: str, command: Callable[[], None]) -> None:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        frame = self.ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew", pady=4)
        frame.columnconfigure(0, weight=1)
        self.ttk.Entry(frame, textvariable=self.values[key]).grid(row=0, column=0, sticky="ew")
        self.ttk.Button(frame, text="浏览…", command=command).grid(row=0, column=1, padx=(6, 0))

    def _browse_ocr(self) -> None:
        from tkinter import filedialog

        value = filedialog.askopenfilename(title="选择 PaddleOCR JSON", filetypes=(("JSON", "*.json"), ("All files", "*.*")))
        if value:
            self.values["ocr"].set(value)

    def _browse_project(self) -> None:
        from tkinter import filedialog

        value = filedialog.askdirectory(title="选择或创建项目目录")
        if value:
            self.values["project"].set(value)

    def _project(self) -> Path:
        value = self.values["project"].get().strip()
        if not value:
            raise ValueError("请选择项目目录")
        return Path(value).expanduser().resolve()

    def _api_environment(self) -> None:
        key = self.values["api_key"].get().strip()
        if key:
            os.environ["DEEPSEEK_API_KEY"] = key
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("请输入 DeepSeek API key，或在启动程序前设置 DEEPSEEK_API_KEY")
        model = self.values["model"].get().strip()
        if model:
            os.environ["DEEPSEEK_MODEL"] = model

    def create_project(self) -> None:
        def action():
            ocr = Path(self.values["ocr"].get().strip()).expanduser().resolve()
            book_id = validate_book_id(self.values["book_id"].get())
            title = self.values["title"].get().strip()
            title_zh = self.values["title_zh"].get().strip()
            if not title or not title_zh:
                raise ValueError("请填写原文书名和中文书名")
            return initialize_project(
                input_json=ocr,
                book_id=book_id,
                book_title=title,
                book_title_zh=title_zh,
                project_dir=self._project(),
                author=self.values["author"].get().strip(),
                source_lang=self.values["source_lang"].get().strip() or "英语",
                target_lang=self.values["target_lang"].get().strip() or "简体中文",
                domain=self.values["domain"].get().strip() or "学术人文社科",
            )

        self._run("初始化项目", action)

    def preflight(self) -> None:
        def action():
            self._api_environment()
            return preflight_project(self._project())

        self._run("零网络预检", action)

    def translate(self) -> None:
        from tkinter import messagebox

        try:
            limit = int(self.values["limit"].get())
            if limit <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("参数错误", "累计翻译数必须是正整数")
            return
        if not messagebox.askyesno("确认翻译", f"将向 DeepSeek 发送文本并可能产生费用。累计目标：{limit}。继续吗？"):
            return

        def action():
            self._api_environment()
            return translate_project(self._project(), target_completed=limit, all_segments=False)

        self._run("翻译", action)

    def open_project(self) -> None:
        from tkinter import messagebox

        try:
            path = self._project()
            path.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("无法打开项目目录", f"{type(exc).__name__}: {exc}")

    def _run(self, name: str, action: Callable[[], Any]) -> None:
        if self.busy:
            self._write("已有任务正在运行，请等待。\n")
            return
        self.busy = True
        self._write(f"\n▶ {name}\n")

        def worker():
            try:
                self.events.put(("result", action()))
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(("done", None))

        threading.Thread(target=worker, daemon=True).start()

    def _write(self, value: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", value)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain_events(self) -> None:
        from tkinter import messagebox

        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "result":
                    self._write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
                elif kind == "error":
                    self._write("✗ " + value + "\n")
                    messagebox.showerror("运行失败", value)
                elif kind == "done":
                    self.busy = False
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)


def main() -> None:
    import tkinter as tk

    root = tk.Tk()
    TranslatorGUI(root)
    root.mainloop()
