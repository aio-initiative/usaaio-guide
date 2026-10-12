"""Writes slides/manifest.csv — one row per deck, ready to paste into a sheet.

The Google Slides column is left blank: it gets filled by
scripts/slides/import_to_drive.gs after the decks are uploaded, or by hand.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_slides as B  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides"
SITE = "https://www.aioinitiative.com"

STAGE = {"round1": "Round 1", "round2": "Round 2", "camp": "Camp", "advanced": "Advanced"}


def main():
    data = B.load_content()
    rows = []
    for module in data["modules"]:
        pptx = OUT / f"{module['id']}.pptx"
        pdf = OUT / f"{module['id']}.pdf"
        if not pptx.exists():
            continue
        try:
            from pptx import Presentation
            count = len(Presentation(str(pptx)).slides._sldIdLst)
        except Exception:
            count = ""
        rows.append({
            "Stage": STAGE.get(module["section"], module["section"]),
            "Code": module.get("code", ""),
            "Module": module["title"],
            "Slides": count,
            "PPTX file": pptx.name,
            "PDF file": pdf.name if pdf.exists() else "",
            "Google Slides link": "",
            "Module page": f"{SITE}/module/?id={module['id']}",
        })

    order = {"Round 1": 0, "Round 2": 1, "Camp": 2, "Advanced": 3}
    rows.sort(key=lambda r: (order.get(r["Stage"], 9), r["Code"]))

    path = OUT / "manifest.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows)} rows -> {path.relative_to(ROOT)}")
    print(f"  total slides: {sum(r['Slides'] for r in rows if r['Slides'])}")


if __name__ == "__main__":
    main()
