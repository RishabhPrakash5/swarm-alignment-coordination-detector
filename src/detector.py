"""Coordination detector: can a monitor flag relay-coordination pages, and how early?

Run from the project root:  python src/detector.py
Writes: results/detector_results.csv, figures/detector_*.png, data/processed/label_check.csv

Population: dse pages with an authors' category (page_family). Positive = 'relay-coordination'.
Tiers (what the monitor can see):
  T1 metadata          features from src/features.py only
  T2 + edit summaries  T1 + TF-IDF of the short edit summaries agents typed
  T3 + full content    T2 + TF-IDF of the text each edit added (reference ceiling: the
                       authors' labels were made from content, so T3 is expected to be high)
Evaluation: time split (train on earliest 70% of pages by creation time, test on the rest),
at four observation windows per page. Metrics: AUROC and average precision (AP), 95% bootstrap CI.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import metadata_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "collusion_wiki"
RES, FIG, PROC = ROOT / "results", ROOT / "figures", ROOT / "data" / "processed"
for d in (RES, FIG, PROC):
    d.mkdir(parents=True, exist_ok=True)

SEED = 0
TRAIN_FRAC = 0.7
WINDOWS = {  # observation window after the page's first edit
    "first edit": None,
    "first hour": pd.Timedelta(hours=1),
    "first day": pd.Timedelta(days=1),
    "full life": pd.Timedelta(days=3650),
}
TIERS = ["T1 metadata", "T2 + edit summaries", "T3 + full content"]


# ---------------------------------------------------------------- data loading
def added_text(body, hunks):
    lines = (body or "").split("\n")
    return "\n".join(
        "\n".join(lines[h["b0"]:h["b1"]]) for h in (hunks or []) if h.get("op") in ("insert", "replace")
    )


def distinct_before(df, key, item):
    """For each row (sorted by time): distinct `item`s seen for this `key` strictly before."""
    first = ~df.duplicated([key, item])
    return first.groupby(df[key]).cumsum() - first.astype(int)


def load():
    rv = pd.read_json(RAW / "revisions.jsonl", lines=True)
    pg = pd.read_json(RAW / "pages.jsonl", lines=True)
    rv["time"] = pd.to_datetime(rv["time"], utc=True)
    rv = rv.sort_values(["time", "page_key", "seq"]).reset_index(drop=True)

    rv["added"] = [added_text(b, h) for b, h in zip(rv["body"], rv["hunks"])]
    rv["added_len"] = rv["added"].str.len()
    rv["n_hunks"] = rv["hunks"].str.len()
    rv["anon"] = rv["label"].eq("")
    rv["secs_since_prev"] = rv.groupby("page_key")["time"].diff().dt.total_seconds()
    rv["is_recreation"] = rv["relation_type"].eq("first_recreation_of")

    named = rv["label"].where(~rv["anon"], "__anon__")
    tmp = rv.assign(label=named)
    rv["acct_prior_edits"] = tmp.groupby("label").cumcount().where(~rv["anon"], 0)
    rv["acct_prior_pages"] = distinct_before(tmp, "label", "page_key").where(~rv["anon"], 0)
    rv["ip16_prior_names"] = distinct_before(tmp, "ip16", "label")

    pages = pg[(pg["wiki"] == "dse") & (pg["page_family_source"] != "none")][["page_key", "page_family", "name"]]
    first = rv.groupby("page_key")["time"].min().rename("created")
    pages = pages.merge(first, left_on="page_key", right_index=True).set_index("page_key")
    pages["y"] = (pages["page_family"] == "relay-coordination").astype(int)
    rv = rv[rv["page_key"].isin(pages.index)]
    return rv, pages


def window(rv, pages, w):
    r = rv.merge(pages[["created"]], left_on="page_key", right_index=True)
    if w is None:
        return r[r.groupby("page_key").cumcount() == 0]
    return r[r["time"] <= r["created"] + w]


# ---------------------------------------------------------------- modelling
def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def joined_text(r, col, index):
    return r.groupby("page_key")[col].apply(lambda s: " ".join(s.dropna().astype(str))).reindex(index).fillna("")


def boot_ci(y, s, fn, n=500):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def fit_score(tier, meta_tr, meta_te, r, tr_idx, te_idx, y_tr):
    sc = StandardScaler().fit(signed_log(meta_tr))
    parts_tr = [sparse.csr_matrix(sc.transform(signed_log(meta_tr)))]
    parts_te = [sparse.csr_matrix(sc.transform(signed_log(meta_te)))]
    text_cols = {"T2 + edit summaries": ["change_summary"], "T3 + full content": ["change_summary", "added"]}
    for col in text_cols.get(tier, []):
        vec = TfidfVectorizer(min_df=3, max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
        parts_tr.append(vec.fit_transform(joined_text(r, col, tr_idx)))
        parts_te.append(vec.transform(joined_text(r, col, te_idx)))
    Xtr, Xte = sparse.hstack(parts_tr).tocsr(), sparse.hstack(parts_te).tocsr()
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(Xtr, y_tr)
    return clf, clf.predict_proba(Xte)[:, 1]


def main():
    rv, pages = load()
    order = pages.sort_values("created")
    cut = order["created"].iloc[int(len(order) * TRAIN_FRAC)]
    tr_idx = order.index[order["created"] < cut]
    te_idx = order.index[order["created"] >= cut]
    y_tr, y_te = pages.loc[tr_idx, "y"].values, pages.loc[te_idx, "y"].values
    print(f"Pages: {len(pages):,} | positives {pages['y'].sum():,} ({pages['y'].mean():.1%})")
    print(f"Time split at {cut}: train {len(tr_idx):,} (pos {y_tr.mean():.1%}) | test {len(te_idx):,} (pos {y_te.mean():.1%})")

    rows, keep = [], {}
    for wname, w in WINDOWS.items():
        r = window(rv, pages, w)
        meta = metadata_features(r).reindex(pages.index).fillna(0)
        meta_tr, meta_te = meta.loc[tr_idx].values, meta.loc[te_idx].values

        # Baselines: chance, and the single most obvious rule ("more than one account edited it")
        base = meta.loc[te_idx, "n_names"].values if "n_names" in meta else np.zeros(len(te_idx))
        for model, s in [("baseline: chance", np.full(len(te_idx), 0.5)), ("baseline: n_names", base)]:
            rows.append(dict(window=wname, model=model, auroc=roc_auc_score(y_te, s) if s.std() else 0.5,
                             ap=average_precision_score(y_te, s) if s.std() else y_te.mean()))
        for tier in TIERS:
            clf, s = fit_score(tier, meta_tr, meta_te, r, tr_idx, te_idx, y_tr)
            a_lo, a_hi = boot_ci(y_te, s, roc_auc_score)
            p_lo, p_hi = boot_ci(y_te, s, average_precision_score)
            rows.append(dict(window=wname, model=tier, auroc=roc_auc_score(y_te, s), auroc_lo=a_lo, auroc_hi=a_hi,
                             ap=average_precision_score(y_te, s), ap_lo=p_lo, ap_hi=p_hi))
            if tier == "T1 metadata":
                keep[wname] = (clf, s, list(meta.columns))
        print(f"  done: {wname}")

    res = pd.DataFrame(rows)
    res.to_csv(RES / "detector_results.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))

    clf, s_full, cols = keep["full life"]
    coef = pd.Series(clf.coef_[0][: len(cols)], index=cols).sort_values()
    print("\nT1 (full life) standardized coefficients:\n" + coef.round(3).to_string())

    # Figure: AP by observation window and tier
    fig, ax = plt.subplots(figsize=(8, 4.5))
    wn = list(WINDOWS)
    for tier, mk in zip(TIERS, "osD"):
        d = res[res["model"] == tier].set_index("window").loc[wn]
        ax.errorbar(wn, d["ap"], yerr=[d["ap"] - d["ap_lo"], d["ap_hi"] - d["ap"]], marker=mk, capsize=3, label=tier)
    d = res[res["model"] == "baseline: n_names"].set_index("window").loc[wn]
    ax.plot(wn, d["ap"], "k--", lw=1, label="baseline: n_names")
    ax.axhline(y_te.mean(), color="grey", lw=1, ls=":", label=f"chance ({y_te.mean():.2f})")
    ax.set_ylabel("average precision (test pages)")
    ax.set_xlabel("how much of each page's life the monitor has seen")
    ax.set_ylim(0, 1)
    ax.set_title("Detecting relay-coordination pages: visibility vs. time")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG / "detector_ap_by_window.png", dpi=150)
    plt.close()

    # Hand-check sheet: 25 top-scored + 25 random test pages (full-life T1 scores)
    te = pages.loc[te_idx].assign(t1_score=s_full)
    top = te.sort_values("t1_score", ascending=False).head(25).assign(group="top_scored")
    rnd = te.drop(top.index).sample(25, random_state=SEED).assign(group="random")
    chk = pd.concat([top, rnd]).sample(frac=1, random_state=SEED)
    out = pd.DataFrame({
        "group": chk["group"],
        "explorer_url": "https://collusion.wiki/explorer/page/" + chk.index,
        "t1_score": chk["t1_score"].round(3),
        "authors_label": chk["page_family"],
        "hand_label (y/n/unclear)": "",
        "notes": "",
    })
    sheet = PROC / "label_check.csv"
    if sheet.exists():  # never overwrite hand labels
        sheet = PROC / "label_check_new.csv"
    out.to_csv(sheet, index=False, encoding="utf-8-sig")
    print("\nWrote results/detector_results.csv, figures/detector_ap_by_window.png, data/processed/label_check.csv")


if __name__ == "__main__":
    main()
