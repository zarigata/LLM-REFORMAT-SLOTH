#!/usr/bin/env python
from __future__ import annotations
import subprocess
import sys

def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()

try:
    log = run(["git", "log", "--pretty=format:%s", "--no-merges", "-n", "50"])
except Exception as e:
    print(f"(changelog) git error: {e}")
    sys.exit(0)

print("\n## Unreleased\n")
for line in log.splitlines():
    print(f"- {line}")
