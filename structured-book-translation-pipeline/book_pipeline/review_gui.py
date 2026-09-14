from __future__ import annotations

"""Evidence-first chapter and glossary review for the desktop application."""

import json
from pathlib import Path
from typing import Any, Callable

from .io_utils import read_jsonl, write_jsonl


STRUCTURE_ACTIONS = ("needs_human", "accept", "reject", "change_level", "change_kind", "merge_with_previous")
STRUCTURE_REASONS = (
    "insufficient_evidence", "same_style", "numbering_parent", "running_header",
    "caption", "body_sentence", "apparatus_repeat", "split_title", "toc_evidence",
    "page_evidence", "other_verified",
)
STRUCTURE_KINDS = ("frontmatter", "part", "part_intro", "chapter", "section", "backmatter", "toc_entry")
GLOSSARY_ACTIONS = ("needs_human", "include", "reject")


def review_files(project: Path, kind: str) -> tuple[Path, Path, Path]:
    if kind == "structure":
        return (
            project / "structure" / "review_packets.jsonl",
            project / "structure_decisions.template.jsonl",
            project / "structure_decisions.jsonl",
        )
    if kind == "glossary":
        return (
            project / "glossary_candidates.jsonl",
            project / "glossary_decisions.template.jsonl",
            project / "glossary_decisions.jsonl",
        )
    raise ValueError(f"Unknown review kind: {kind}")


def load_review(project: Path, kind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    candidates_path, template_path, decisions_path = review_files(project, kind)
    if not candidates_path.is_file():
        raise FileNotFoundError("请先运行‘准备/检查’，生成待审核项")
    candidates = list(read_jsonl(candidates_path))
    if not candidates:
        return [], [], decisions_path
    if not template_path.is_file():
        raise FileNotFoundError("审核模板不存在，请重新运行‘准备/检查’")
    template = list(read_jsonl(template_path))
    saved = list(read_jsonl(decisions_path)) if decisions_path.exists() else template
    ids = [str(row.get("candidate_id", "")) for row in candidates]
    if len(ids) != len(set(ids)) or set(ids) != {row.get("candidate_id") for row in template}:
        raise ValueError("候选项与模板不一致，请重新运行准备/检查")
    # A stale decision file may refer to a previous candidate generation.
    if len(saved) != len(ids) or {row.get("candidate_id") for row in saved} != set(ids):
        saved = template
    by_id = {row["candidate_id"]: row for row in saved}
    return candidates, [dict(by_id[candidate_id]) for candidate_id in ids], decisions_path


def decision_from_fields(kind: str, candidate_id: str, fields: dict[str, str]) -> dict[str, Any]:
    action = fields["decision"].strip()
    if kind == "structure":
        if action not in STRUCTURE_ACTIONS:
            raise ValueError("请选择有效的章节裁决")
        reason = fields["reason"].strip()
        if reason not in STRUCTURE_REASONS:
            raise ValueError("请选择证据原因")
        evidence = [item.strip() for item in fields["evidence"].splitlines() if item.strip()]
        if not evidence:
            raise ValueError("章节裁决必须填写至少一个真实证据引用，每行一个")
        row: dict[str, Any] = {
            "candidate_id": candidate_id, "decision": action,
            "reason_code": reason, "evidence_refs": evidence,
        }
        if action == "change_level":
            try:
                level = int(fields["level"].strip())
            except ValueError as exc:
                raise ValueError("目标层级必须是 2 到 6 的整数") from exc
            if level not in range(2, 7):
                raise ValueError("目标层级必须是 2 到 6 的整数")
            row["level"] = level
        if action == "change_kind":
            value = fields["kind"].strip()
            if value not in STRUCTURE_KINDS:
                raise ValueError("请选择有效的目标类型")
            row["kind"] = value
        if action == "merge_with_previous":
            value = fields["merge_with_node_id"].strip()
            if not value.startswith("node-"):
                raise ValueError("合并目标必须是当前证据中的 node-... ID")
            row["merge_with_node_id"] = value
        return row
    if kind == "glossary":
        if action not in GLOSSARY_ACTIONS:
            raise ValueError("请选择有效的术语裁决")
        row = {"candidate_id": candidate_id, "decision": action}
        if action == "include":
            translation = fields["translation"].strip()
            evidence = [item.strip() for item in fields["evidence"].splitlines() if item.strip()]
            if not translation or not evidence:
                raise ValueError("收录术语必须填写译名和至少一条证据")
            row.update({"translation": translation, "evidence": evidence})
            if fields["category"].strip():
                row["category"] = fields["category"].strip()
        if action == "reject":
            reason = fields["reason"].strip()
            if not reason:
                raise ValueError("拒绝术语必须填写原因")
            row["reason"] = reason
        return row
    raise ValueError(f"Unknown review kind: {kind}")


class ReviewWindow:
    def __init__(self, parent: Any, project: Path, kind: str, run_action: Callable[[str, Callable[[], Any]], None]):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.project = project
        self.kind = kind
        self.run_action = run_action
        self.candidates, self.rows, self.decisions_path = load_review(project, kind)
        self.active = -1
        self.window = tk.Toplevel(parent)
        self.window.title("章节结构审核" if kind == "structure" else "术语审核")
        self.window.geometry("1050x730")
        self.window.minsize(800, 600)
        self.window.transient(parent)
        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)
        ttk.Label(frame, text="逐项阅读原始证据并裁决；未解决项保持 needs_human。保存后才可编译。")\
            .grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.listbox = tk.Listbox(frame, width=30, exportselection=False)
        self.listbox.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        for candidate in self.candidates:
            label = str(candidate.get("term") or candidate.get("exact_text") or candidate.get("title") or candidate["candidate_id"])
            self.listbox.insert("end", label[:55])
        self.listbox.bind("<<ListboxSelect>>", self._select)
        right = ttk.Frame(frame)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(1, weight=1)
        right.rowconfigure(0, weight=1)
        self.evidence_view = tk.Text(right, height=17, wrap="word")
        self.evidence_view.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        self.evidence_view.configure(state="disabled")
        self.fields: dict[str, Any] = {}
        self.fields["decision"] = tk.StringVar()
        self.fields["reason"] = tk.StringVar()
        self.fields["translation"] = tk.StringVar()
        self.fields["category"] = tk.StringVar()
        self.fields["level"] = tk.StringVar()
        self.fields["kind"] = tk.StringVar()
        self.fields["merge_with_node_id"] = tk.StringVar()
        ttk.Label(right, text="裁决").grid(row=1, column=0, sticky="w")
        ttk.Combobox(right, textvariable=self.fields["decision"], state="readonly",
                     values=STRUCTURE_ACTIONS if kind == "structure" else GLOSSARY_ACTIONS)\
            .grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(right, text="原因代码" if kind == "structure" else "拒绝原因").grid(row=2, column=0, sticky="w")
        if kind == "structure":
            ttk.Combobox(right, textvariable=self.fields["reason"], state="readonly", values=STRUCTURE_REASONS)\
                .grid(row=2, column=1, sticky="ew", pady=3)
        else:
            ttk.Entry(right, textvariable=self.fields["reason"]).grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Label(right, text="证据引用（每行一条）").grid(row=3, column=0, sticky="nw")
        self.evidence_input = tk.Text(right, height=4, wrap="word")
        self.evidence_input.grid(row=3, column=1, sticky="ew", pady=3)
        extras = (("translation", "译名"), ("category", "类别")) if kind == "glossary" else (
            ("level", "目标层级 2–6"), ("kind", "目标类型"), ("merge_with_node_id", "合并目标 node ID"))
        for offset, (key, label) in enumerate(extras, 4):
            ttk.Label(right, text=label).grid(row=offset, column=0, sticky="w")
            if key == "kind":
                ttk.Combobox(right, textvariable=self.fields[key], values=STRUCTURE_KINDS)\
                    .grid(row=offset, column=1, sticky="ew", pady=3)
            else:
                ttk.Entry(right, textvariable=self.fields[key]).grid(row=offset, column=1, sticky="ew", pady=3)
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        save_button = ttk.Button(buttons, text="保存审核决定", command=self.save)
        save_button.pack(side="left", padx=(0, 8))
        compile_button = ttk.Button(buttons, text="编译决定并重新准备", command=self.compile)
        compile_button.pack(side="left")
        if self.candidates:
            self.listbox.selection_set(0)
            self._show(0)
        else:
            save_button.configure(state="disabled")
            compile_button.configure(state="disabled")
            self.evidence_view.configure(state="normal")
            self.evidence_view.insert("end", "当前没有待审核项。")
            self.evidence_view.configure(state="disabled")

    def _show(self, index: int) -> None:
        self.active = index
        self.evidence_view.configure(state="normal")
        self.evidence_view.delete("1.0", "end")
        self.evidence_view.insert("end", json.dumps(self.candidates[index], ensure_ascii=False, indent=2))
        self.evidence_view.configure(state="disabled")
        row = self.rows[index]
        for key in self.fields:
            source_key = "reason_code" if key == "reason" and self.kind == "structure" else key
            self.fields[key].set(str(row.get(source_key, "")))
        evidence_key = "evidence_refs" if self.kind == "structure" else "evidence"
        self.evidence_input.delete("1.0", "end")
        self.evidence_input.insert("end", "\n".join(row.get(evidence_key, [])))

    def _current(self) -> dict[str, Any]:
        fields = {key: value.get() for key, value in self.fields.items()}
        fields["evidence"] = self.evidence_input.get("1.0", "end").strip()
        return decision_from_fields(self.kind, self.rows[self.active]["candidate_id"], fields)

    def _select(self, _event: Any) -> None:
        from tkinter import messagebox

        selected = self.listbox.curselection()
        if not selected or selected[0] == self.active:
            return
        if self.active >= 0:
            try:
                self.rows[self.active] = self._current()
            except ValueError as exc:
                messagebox.showerror("审核项尚不完整", str(exc), parent=self.window)
                self.listbox.selection_clear(0, "end")
                self.listbox.selection_set(self.active)
                return
        self._show(selected[0])

    def save(self) -> bool:
        from tkinter import messagebox

        try:
            if self.active >= 0:
                self.rows[self.active] = self._current()
            write_jsonl(self.decisions_path, self.rows)
        except (ValueError, OSError) as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.window)
            return False
        messagebox.showinfo("已保存", str(self.decisions_path), parent=self.window)
        return True

    def compile(self) -> None:
        if not self.save():
            return
        from .workflow import compile_project_glossary, compile_reviews, prepare_project

        def action() -> dict[str, Any]:
            if self.kind == "structure":
                compiled = compile_reviews(self.project, self.decisions_path)
            else:
                compiled = compile_project_glossary(self.project, self.decisions_path)
            return {"compiled": compiled, "prepared": prepare_project(self.project)}

        self.run_action("编译审核并重新准备", action)
