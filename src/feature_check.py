"""Each T1 feature alone: AUROC and medians on TRAINING pages, full-life window. Writes results/feature_alone_train.csv."""
import sys
sys.path.insert(0, r"D:\Acad\Endeavours\Swarm Alignment Hackathon\src")
import pandas as pd
from sklearn.metrics import roc_auc_score
from detector import load, window, WINDOWS, TRAIN_FRAC
from features import metadata_features
rv, pages = load()
order = pages.sort_values("created")
cut = order["created"].iloc[int(len(order) * TRAIN_FRAC)]
tr = order.index[order["created"] < cut]
meta = metadata_features(window(rv, pages, WINDOWS["full life"])).reindex(pages.index).fillna(0).loc[tr]
y = pages.loc[tr, "y"]
out = pd.DataFrame({
    "auroc_alone_train": {c: roc_auc_score(y, meta[c]) for c in meta},
    "median_relay": meta[y == 1].median(),
    "median_other": meta[y == 0].median(),
}).sort_values("auroc_alone_train")
print(out.round(3).to_string()); out.to_csv(__import__("pathlib").Path(__file__).resolve().parents[1] / "results" / "feature_alone_train.csv")
