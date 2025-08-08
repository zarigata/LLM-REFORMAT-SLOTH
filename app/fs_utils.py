from __future__ import annotations
import os
from typing import List, Dict


def list_dir_tree(path: str, max_items: int = 500) -> List[Dict[str, str]]:
    """Return a flat list of files/dirs under path with limited depth and count."""
    rows: List[Dict[str, str]] = []
    count = 0
    for root, dirs, files in os.walk(path):
        dirs.sort()
        files.sort()
        for d in dirs:
            rel = os.path.relpath(os.path.join(root, d), path)
            rows.append({"type": "dir", "path": rel})
            count += 1
            if count >= max_items:
                return rows
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), path)
            rows.append({"type": "file", "path": rel})
            count += 1
            if count >= max_items:
                return rows
    return rows
