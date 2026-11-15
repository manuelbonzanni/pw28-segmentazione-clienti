"""Genera i dati sintetici di CasaPronta"""
import argparse
import numpy as np
import pandas as pd

import config as C

ARCHETYPES, CAT_PREF, ZONE_P = C.ARCHETYPES, C.CAT_PREF, C.ZONE_P  # parametri del generatore in config.py


def generate(n=C.N_CUSTOMERS, seed=C.SEED, missing_rate=0.03):
    rng = np.random.default_rng(seed)
    ref = pd.Timestamp(C.REFERENCE_DATE)
    names = list(ARCHETYPES)
    counts = rng.multinomial(n, [ARCHETYPES[a]["share"] for a in names])
    rows = []
    for name, cnt in zip(names, counts):
        a = ARCHETYPES[name]
        for _ in range(cnt):
            tenure = int(rng.integers(*a["ten"]))
            n_req = int(rng.poisson(a["lam"])) if a["lam"] > 0 else 0
            if name != "inattivi":
                n_req = max(n_req, 1)                   # chi non è "inattivo" ha ordinato almeno una volta
            if name == "inattivi":
                n_req = 0                               # nessuna richiesta nella finestra di 90 giorni
                rec = int(rng.integers(*a["rec"]))
            else:
                rec = int(rng.integers(a["rec"][0], a["rec"][1] + 1))
            rec = min(rec, tenure)                      # non si può ordinare prima di iscriversi
            if n_req > 0:
                rec = min(rec, C.WINDOW_DAYS)           # coerenza: richieste in finestra => recency <= 90
            # importo totale = somma di n ticket lognormali (dispersione plausibile degli importi)
            mu = np.log(a["ticket"]) - a["sd"] ** 2 / 2
            tickets = rng.lognormal(mu, a["sd"], size=n_req)
            plan = rng.choice(C.PLANS, p=_plan_probs(a))
            rows.append(dict(
                zona=rng.choice(C.ZONES, p=ZONE_P),
                data_iscrizione=(ref - pd.Timedelta(days=tenure)).date().isoformat(),
                richieste_90g=n_req,
                importo_totale_90g=round(float(tickets.sum()), 2),
                giorni_ultima_richiesta=rec,
                categoria_preferita=rng.choice(C.CATEGORIES, p=CAT_PREF[name]),
                piano=plan,
                _profilo_latente=name,
            ))
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "customer_id", [f"C{i + 1:04d}" for i in range(len(df))])
    # rumore realistico -> alcuni valori mancanti (errori di tracciamento)
    for col in ("importo_totale_90g", "giorni_ultima_richiesta"):
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan
    return df


def _plan_probs(a):
    p_trial, p_plus = a["trial"], a["plus"]
    return [max(0.0, 1 - p_plus - p_trial), p_plus, p_trial]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=C.N_CUSTOMERS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    df = generate(args.n, args.seed)
    C.DATA_DIR.mkdir(exist_ok=True)
    df.drop(columns="_profilo_latente").to_csv(C.DATA_DIR / "clienti.csv", index=False)
    df[["customer_id", "_profilo_latente"]].to_csv(C.DATA_DIR / "profilo_latente_verita.csv", index=False)
    print(f"Generati {len(df)} clienti -> data/clienti.csv")
    print(df["_profilo_latente"].value_counts().to_string())


if __name__ == "__main__":
    main()
