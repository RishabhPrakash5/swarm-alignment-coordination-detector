"""Compare hand labels (data/processed/label_check.csv) with the authors' labels and the T1 detector.
Writes results/label_agreement.txt."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "data/processed/label_check.csv")
d["mine"] = d["hand_label (y/n/unclear)"].str.strip().str.lower()
d["kind"] = d["explicit/implicit"].fillna("")
d["authors_relay"] = d["authors_label"].eq("relay-coordination")

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

out = []
def line(s=""):
    out.append(s); print(s)

line(f"Hand-labeled pages: {len(d)} (top-scored {d.group.eq('top_scored').sum()}, random {d.group.eq('random').sum()})")
line("Hand labels: " + d["mine"].value_counts().to_dict().__repr__())
line("Kinds among y: " + d.loc[d.mine.eq('y'), 'kind'].value_counts().to_dict().__repr__())
line("\nCross-tab, authors' label (relay vs other) x hand label:")
line(pd.crosstab(d["authors_relay"].map({True: "authors: relay", False: "authors: other"}), d["mine"], margins=True).to_string())

dec = d[d.mine.isin(["y", "n"])]
agree = (dec.mine.eq("y") == dec.authors_relay)
lo, hi = wilson(agree.sum(), len(dec))
line(f"\nAgreement on decided pages (unclear dropped): {agree.sum()}/{len(dec)} = {agree.mean():.0%} (95% Wilson {lo:.0%}-{hi:.0%})")
for name, sub in [("authors' relay pages", dec[dec.authors_relay]), ("authors' other pages", dec[~dec.authors_relay])]:
    k = sub.mine.eq("y").sum(); lo, hi = wilson(k, len(sub))
    line(f"  hand 'y' among {name}: {k}/{len(sub)} = {k/len(sub):.0%} (95% {lo:.0%}-{hi:.0%})")

line("\nT1 detector precision (share of pages that are coordination):")
for g in ["top_scored", "random"]:
    sub = d[d.group.eq(g)]
    for lab, hit in [("authors' relay", sub.authors_relay), ("hand y (unclear counted as not)", sub.mine.eq("y"))]:
        k = hit.sum(); lo, hi = wilson(k, len(sub))
        line(f"  {g:10s} by {lab:32s}: {k}/{len(sub)} = {k/len(sub):.0%} (95% {lo:.0%}-{hi:.0%})")
(ROOT / "results/label_agreement.txt").write_text("\n".join(out), encoding="utf-8")
