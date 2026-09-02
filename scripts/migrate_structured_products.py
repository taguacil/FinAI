#!/usr/bin/env python3
"""One-off migration: reclassify structured products from `bond`.

Reverse convertibles, credit-linked notes (CLNs), callable/autocallable notes,
shark notes and similar bank-issued structured products were historically
booked as `bond`. Financially they are structured products (debt wrappers with
embedded equity/credit options), so they belong in the new
``InstrumentType.STRUCTURED_PRODUCT`` class.

The classifier is name-based and conservative: genuine senior/government bonds
(incl. "Treasury Notes ... Senior") stay as bonds; only names that clearly
denote a structured wrapper are moved. Run with --apply to write changes
(a .pre_struct_migration backup is written per file first); default is a dry run.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
from pathlib import Path

# Clear structured-product markers -> always structured.
STRONG = re.compile(
    r"reverse convertible|\bRC\b|credit linked note|\bCLN\b|autocallable"
    r"|\borion\b|shark note|tracker certificate|best performer",
    re.I,
)
# Plain senior / government debt -> stays a bond even if "note(s)" appears
# (e.g. "Etats-Unis Treasury Notes Senior").
GOVT_SENIOR = re.compile(r"treasury|senior", re.I)
# Weaker structured hints -> structured only when not senior/government.
WEAK = re.compile(r"\bnotes?\b|callable", re.I)


def classify(name: str | None) -> str:
    n = name or ""
    if STRONG.search(n):
        return "structured_product"
    if GOVT_SENIOR.search(n):
        return "bond"
    if WEAK.search(n):
        return "structured_product"
    return "bond"


def migrate_file(path: Path, apply: bool) -> int:
    data = json.loads(path.read_text())

    # 1) Decide which symbols become structured, from every bond-typed instrument
    #    dict across positions and transactions.
    struct_symbols: set[str] = set()

    def scan(instr: dict) -> None:
        if instr.get("instrument_type") == "bond":
            if classify(instr.get("name")) == "structured_product":
                struct_symbols.add(instr.get("symbol"))

    for pos in (data.get("positions") or {}).values():
        scan(pos.get("instrument", {}))
    for txn in (data.get("transactions") or []):
        scan(txn.get("instrument", {}))

    if not struct_symbols:
        return 0

    # 2) Rewrite every bond-typed instrument dict for those symbols.
    changed = 0

    def rewrite(instr: dict) -> None:
        nonlocal changed
        if (instr.get("instrument_type") == "bond"
                and instr.get("symbol") in struct_symbols):
            instr["instrument_type"] = "structured_product"
            changed += 1

    for pos in (data.get("positions") or {}).values():
        rewrite(pos.get("instrument", {}))
    for txn in (data.get("transactions") or []):
        rewrite(txn.get("instrument", {}))

    print(f"{path.name}: {len(struct_symbols)} symbol(s) -> structured_product "
          f"({changed} instrument records)")
    for s in sorted(struct_symbols):
        print(f"    {s}")

    if apply and changed:
        shutil.copy2(path, path.with_suffix(path.suffix + ".pre_struct_migration"))
        path.write_text(json.dumps(data, indent=2))

    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--dir", default="data/portfolios")
    args = ap.parse_args()

    total = 0
    for p in sorted(glob.glob(f"{args.dir}/*.json")):
        total += migrate_file(Path(p), args.apply)

    mode = "APPLIED" if args.apply else "DRY RUN (use --apply to write)"
    print(f"\n{mode}: {total} instrument record(s) reclassified.")


if __name__ == "__main__":
    main()
