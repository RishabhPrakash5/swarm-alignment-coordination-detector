"""Metadata features (Tier 1) for the coordination detector.

A Tier 1 monitor sees no page text, no page titles and no account-name strings.
It only sees who / when / how many / how big for each edit.

`r` = all edits of all pages, already cut to the observation window
(first edit / first hour / first day / full life). One row per edit. Allowed columns:

  page_key            which page (group by this)
  time                edit time (UTC Timestamp)
  seq                 edit number on the page (1 = page creation)
  label               account name — use only as an ID (count, compare), never read the string
  anon                True if the edit was saved with no account name
  ip16                first two IP octets — use only as an ID
  body_len, lines     size of the full page after the edit
  added_len           characters this edit added
  n_hunks             number of separate changed regions in this edit
  secs_since_prev     seconds since the previous edit on the same page (NaN for the first edit)
  acct_prior_edits    edits this account made anywhere BEFORE this edit
  acct_prior_pages    distinct pages this account edited BEFORE this edit
  ip16_prior_names    distinct account names seen on this IP prefix BEFORE this edit
  is_recreation       edit re-creates a page the admin had deleted

All "prior" columns only use the past, so they are safe for a real-time monitor.
NOT allowed here: body, added, change_summary, name/page title (those are Tier 2/3 or label leaks).

Return: DataFrame indexed by page_key, one numeric column per feature.
Counts get log-scaled and standardized automatically.
"""
import pandas as pd
import numpy as np


def metadata_features(r: pd.DataFrame) -> pd.DataFrame:
    r = r.sort_values(["page_key", "time", "seq"])
    g = r.groupby("page_key")
    f = pd.DataFrame(index=g.size().index)
    pairs = (g.size() - 1).clip(lower=1)          # number of consecutive edit pairs

    # Basic counts
    f["n_edits"] = g.size()
    f["n_names"] = g["label"].nunique()
    f["n_ip16"] = g["ip16"].nunique()

    # Scenario 1: short "anyone here?" pings first, longer Q&A later; a handful of names do most edits
    first_len = g["added_len"].first()
    later_len = r[g.cumcount() > 0].groupby("page_key")["added_len"].mean()
    f["len_growth"] = (np.log1p(later_len) - np.log1p(first_len)).reindex(f.index)
    f["edits_per_name"] = f["n_edits"] / f["n_names"].clip(lower=1)

    # Scenario 2: readers drop one short note ("thanks") and leave
    per_name = r.groupby(["page_key", "label"]).size()
    f["frac_oneoff_names"] = (per_name == 1).groupby("page_key").mean()
    f["median_added_len"] = g["added_len"].median()

    # Scenario 3: early discoverers are regulars active on many pages
    f["median_prior_pages"] = g["acct_prior_pages"].median()
    f["frac_fresh_acct"] = (r["acct_prior_edits"] == 0).groupby(r["page_key"]).mean()

    # Scenario 4: two or three accounts chatting -> turn-taking, minutes apart
    prev = g["label"].shift()
    switch = prev.notna() & (r["label"] != prev)
    f["turn_switch"] = switch.groupby(r["page_key"]).sum() / pairs
    f["quick_reply_share"] = (switch & (r["secs_since_prev"] <= 300)).groupby(r["page_key"]).sum() / pairs

    return f.fillna(0)
