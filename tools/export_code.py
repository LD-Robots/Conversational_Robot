#!/usr/bin/env python3
# Exportă DOAR codul .py din src/ și tools/ într-un singur fișier: CODE_ONLY.txt

from pathlib import Path

ROOT = Path("/home/delia/Conversational_Robot/Conversational_Bot")
OUT  = ROOT / "CODE_ONLY.txt"

def collect(paths):
    files = []
    for p in paths:
        if p.is_dir():
            files += sorted(p.rglob("*.py"))
    return files

def main():
    src = ROOT / "src"
    tools = ROOT / "tools"

    candidates = collect([src, tools])
    if not candidates:
        print("[E] Nu am găsit fișiere .py în src/ sau tools/")
        return

    with OUT.open("w", encoding="utf-8", errors="ignore") as w:
        w.write("# === CODE ONLY export (.py din src/ și tools/) ===\n")
        w.write(f"# Root: {ROOT}\n")
        w.write(f"# Files: {len(candidates)}\n\n")
        for p in candidates:
            rel = p.relative_to(ROOT)
            w.write(f"\n# ===== FILE: {rel} =====\n")
            w.write("```python\n")
            try:
                w.write(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception as e:
                w.write(f"<<EROARE LA CITIRE: {e}>>")
            w.write("\n```\n")

    print(f"[OK] Export complet: {OUT} ({len(candidates)} fișiere)")

if __name__ == "__main__":
    main()
