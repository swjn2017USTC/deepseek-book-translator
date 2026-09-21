"""Safe native EPUB inspection, segmentation, rewrite and repackaging."""

from .inventory import inspect_epub, inspect_tree, write_inspection_reports
from .repack import pack_epub
from .render import render_epub
from .rewrite import EPUBRewriteError, rewrite_epub_tree
from .segments import apply_segment_slots, reconstruct_segment_slots, segment_epub
from .validate import validate_rendered_tree

__all__ = [
    "inspect_epub",
    "inspect_tree",
    "write_inspection_reports",
    "segment_epub",
    "reconstruct_segment_slots",
    "apply_segment_slots",
    "rewrite_epub_tree",
    "EPUBRewriteError",
    "validate_rendered_tree",
    "pack_epub",
    "render_epub",
]
