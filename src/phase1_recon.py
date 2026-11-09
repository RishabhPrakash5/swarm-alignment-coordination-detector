"""Phase 1 recon: basic counts, timelines, IP-prefix stats, first-pass signature check.

Run from the project root:  python src/phase1_recon.py
Writes: data/processed/revisions_added.parquet-like CSV, figures/*.png, prints a summary.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "collusion_wiki"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

rv = pd.read_json(RAW / "revisions.jsonl", lines=True)
ev = pd.read_json(RAW / "events.jsonl", lines=True)
pg = pd.read_json(RAW / "pages.jsonl", lines=True)
lb = pd.read_json(RAW / "labels.jsonl", lines=True)
rv["time"] = pd.to_datetime(rv["time"], utc=True)
rv["day"] = rv["time"].dt.date


# --- Text added by each edit -------------------------------------------------
# Each revision stores the WHOLE page after the save. `hunks` give line ranges
# (b0..b1) in the new body that this edit inserted or replaced.
def added_text(row):
    lines = (row["body"] or "").split("\n")
    parts = [
        "\n".join(lines[h["b0"]:h["b1"]])
        for h in (row["hunks"] or [])
        if h.get("op") in ("insert", "replace")
    ]
    return "\n".join(parts)


rv["added"] = rv.apply(added_text, axis=1)

# First-pass signature regex: "-- Name" at the end of a line. CHECK BY EYE in Phase 2.
SIG = re.compile(r"(?:^|\s)--\s*([A-Za-z][A-Za-z0-9_\-]{2,60})\s*$", re.M)
rv["sigs"] = rv["added"].apply(lambda t: SIG.findall(t))
rv["sig_last"] = rv["sigs"].apply(lambda s: s[-1] if s else None)
has_sig = rv["sig_last"].notna()
mismatch = has_sig & (rv["sig_last"].str.lower() != rv["label"].str.lower())

# --- Summary -----------------------------------------------------------------
edits_per_name = rv.groupby("label").size()
ip_per_name = rv.groupby("label")["ip16"].nunique()
names_per_ip = rv.groupby("ip16")["label"].nunique()
sig_names = set(rv.loc[has_sig, "sig_last"].str.lower())
acct_names = set(rv["label"].str.lower())

lines = []
P = lines.append
P(f"Revisions: {len(rv):,} | pages: {len(pg):,} | names: {len(lb):,} | events: {len(ev):,}")
P(f"Edits per wiki: {rv['wiki'].value_counts().to_dict()}")
P(f"Event types: {ev['event_type'].value_counts().to_dict()}")
P(f"Human-flagged handles: {lb.loc[lb['is_human_handle'], 'label'].tolist()}")
P(f"Date range: {rv['time'].min()} -> {rv['time'].max()}")
P("")
P(f"Edits per name: median {edits_per_name.median():.0f}, mean {edits_per_name.mean():.1f}, "
  f"max {edits_per_name.max()} ; names with exactly 1 edit: {(edits_per_name == 1).mean():.1%}")
P(f"IP /16 prefixes per name: median {ip_per_name.median():.0f}, mean {ip_per_name.mean():.2f}, "
  f"max {ip_per_name.max()} ; names seen on >1 prefix: {(ip_per_name > 1).mean():.1%}")
P(f"Names per IP /16 prefix: median {names_per_ip.median():.0f}, mean {names_per_ip.mean():.1f}, "
  f"max {names_per_ip.max()} ; distinct prefixes: {len(names_per_ip):,}")
P("")
P(f"Edits whose ADDED text has a '-- Name' signature: {has_sig.mean():.1%} ({has_sig.sum():,})")
P(f"  ...of those, signature != account name: {mismatch.sum() / max(has_sig.sum(), 1):.1%} ({mismatch.sum():,})")
P(f"Distinct signature names: {len(sig_names):,} ; also used as an account: {len(sig_names & acct_names):,} ; "
  f"signature-only names: {len(sig_names - acct_names):,}")
sig_per_acct = rv[has_sig].groupby("label")["sig_last"].nunique()
acct_per_sig = rv[has_sig].assign(s=rv["sig_last"].str.lower()).groupby("s")["label"].nunique()
P(f"Signatures per account (accounts that sign): median {sig_per_acct.median():.0f}, max {sig_per_acct.max()} ; "
  f">1 signature: {(sig_per_acct > 1).mean():.1%}")
P(f"Accounts per signature: median {acct_per_sig.median():.0f}, max {acct_per_sig.max()} ; "
  f">1 account: {(acct_per_sig > 1).mean():.1%}")
P("Top signatures by number of accounts posting them:")
for s, n in acct_per_sig.sort_values(ascending=False).head(8).items():
    P(f"   {s}: {n} accounts")

summary = "\n".join(lines)
print(summary)
(OUT / "phase1_summary.txt").write_text(summary, encoding="utf-8")

# --- Save processed table for Phase 2 -----------------------------------------
cols = ["rev_id", "page_key", "wiki", "label", "ip16", "time", "seq", "sig_last", "added"]
rv[cols].assign(sigs=rv["sigs"].str.join("|")).to_csv(OUT / "revisions_added.csv", index=False)

# --- Figures ------------------------------------------------------------------
daily = rv.groupby(["day", "wiki"]).size().unstack(fill_value=0)
ax = daily.plot(kind="bar", stacked=True, figsize=(12, 4), width=0.9)
ax.set_yscale("log")
ax.set_ylabel("edits per day (log)")
ax.set_title("Agent edits per day by wiki (sanity check vs collusion.wiki explorer)")
plt.tight_layout()
plt.savefig(FIG / "p1_daily_edits.png", dpi=150)
plt.close()

first_seen = rv.groupby("label")["time"].min().dt.date.value_counts().sort_index()
ax = first_seen.plot(kind="bar", figsize=(12, 4), width=0.9, color="tab:orange")
ax.set_ylabel("new names first seen")
ax.set_title("New agent names per day")
plt.tight_layout()
plt.savefig(FIG / "p1_new_names_per_day.png", dpi=150)
plt.close()
print("\nWrote data/processed/phase1_summary.txt, revisions_added.csv, figures/p1_*.png")
