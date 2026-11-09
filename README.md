# Who is talking on the wall?

Submission for the Swarm Alignment Hackathon 2026 (Apart Research, Track 4: oversight and monitoring).

In mid 2026 AI agents doing timed web lookup tasks found they could write to small public wikis and used them as a message board. This project asks a simple question. If you wanted to catch that coordination while it was happening, what would your monitor need to see, and how early would it work?

Think of students in separate exam cubicles who discover they can leave notes on the bathroom wall. A monitor could see only the visit log (who came in, when, how much they wrote), or also the short notes on the door, or the whole wall. I compare those three levels.

## Setup

- Data: the [collusion.wiki](https://collusion.wiki/explorer/download) export, 3,908 pages on the German wiki dse
- Target: the 709 pages the dataset authors labeled `relay-coordination`
- Tiers: T1 metadata only, T2 adds the edit summaries agents typed, T3 adds full content
- Windows: after the first edit, first hour, first day and full life of each page
- Split by time: train on pages created before 18 June 2026 20:10 UTC, test on the rest
- Model: logistic regression, scored with AUROC and average precision plus bootstrap 95% CIs
- Hand check: 50 test pages labeled by me from their content

## Results

| Window | T1 metadata | T2 + summaries | T3 + content |
|---|---|---|---|
| first edit | 0.59 | 0.68 | 0.82 |
| first hour | 0.63 | 0.71 | 0.84 |
| full life | 0.65 | 0.73 | 0.85 |

Test AUROC. T3 is inflated because the labels were made from content.

- What the monitor can see matters more than how long it watches.
- Turn taking between accounts is the strongest metadata signal. My other ideas (thank you notes, messages growing over time, early adopters being regulars) did not hold up.
- 21 of the 25 pages the metadata monitor ranked highest show coordination by hand, against 4 of 25 random pages.
- The authors' relay label is precise but misses a lot. Most coordination I found was implicit, with agents revising a shared link list instead of messaging each other.

## Run it

```bash
pip install -r requirements.txt
```

Download the collusion.wiki export and unzip it into `data/raw/collusion_wiki/`. Then:

```bash
python src/detector.py
python src/feature_check.py
python src/label_agreement.py
```

| File | What it does |
|---|---|
| `src/features.py` | Tier 1 metadata features |
| `src/detector.py` | Loads data, trains and scores all tiers and windows, makes the figure |
| `src/feature_check.py` | Scores each feature on its own |
| `src/label_agreement.py` | Compares my hand labels with the authors' labels and the detector |
| `src/phase1_recon.py` | First look at the data |
| `data/processed/label_check.csv` | My 50 hand labels |

The raw data is not in this repo. Get it from collusion.wiki.
