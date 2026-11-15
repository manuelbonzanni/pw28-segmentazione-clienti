"""Valutazione completa: scrive CSV e metriche in results/"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import silhouette_score

import config as C
import generate_data as G
import newclients as N
import segmentation as S


def stability(X, k, n_runs=30, n_init=C.N_INIT, init="k-means++"):
    """ARI medio e minimo tra partizioni ottenute con semi diversi
    Con n_init=20 e k-means++ (configurazione adottata) il valore 1,0 è atteso; la stessa misura con
    n_init=1 e init="random" mostra quanto contribuiscono k-means++ e le ripetizioni alla riproducibilità"""
    from sklearn.cluster import KMeans
    labs = [KMeans(n_clusters=k, init=init, n_init=n_init, random_state=s).fit_predict(X) for s in range(n_runs)]
    vals = [ari(labs[i], labs[j]) for i in range(n_runs) for j in range(i + 1, n_runs)]
    return float(np.mean(vals)), float(np.min(vals))


def latent_profiles(df):
    """Profilo latente di ogni riga di df, allineato per customer_id (non per posizione)"""
    lat = pd.read_csv(C.DATA_DIR / "profilo_latente_verita.csv").set_index("customer_id")
    return lat.loc[df["customer_id"], "_profilo_latente"].to_numpy()


def bootstrap_stability(df, k, n_boot=30, seed=0):
    """ARI tra il modello completo e modelli addestrati su campioni bootstrap"""
    full = S.run_segmentation(df, k=k)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        m = S.run_segmentation(df.iloc[idx].reset_index(drop=True), k=k, seed=int(rng.integers(1e6)))
        lab, _ = m.predict(df)
        vals.append(ari(full.labels, lab))
    return float(np.mean(vals)), float(np.std(vals))


def ablation(df, truth, ks=(4, 5)):
    """Ablazione: confronta variabili e trasformazioni a k fisso, rispetto ai profili latenti (truth allineato a df)"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    d = df.copy()
    d["importo_medio"] = np.where(d["richieste_90g"] > 0, d["importo_totale_90g"] / d["richieste_90g"].clip(lower=1), 0.0)
    F, L = list(C.FEATURES), list(C.LOG_FEATURES)   # F = RFM + anzianità (configurazione adottata)
    R = list(C.RFM_FEATURES)
    configs = {"RFM + anzianità con log + z-score (adottata)": (F, L),
               "Solo RFM con log + z-score": (R, L),
               "RFM + anzianità solo z-score (senza log)": (F, []),
               "RFM + anzianità + importo medio": (F + ["importo_medio"], L + ["importo_medio"])}
    rows = []
    for name, (cols, logc) in configs.items():
        X = d[cols].astype(float).copy()
        for c in logc:
            X[c] = np.log1p(X[c])
        Z = StandardScaler().fit_transform(X)
        for k in ks:
            lab = KMeans(k, n_init=C.N_INIT, random_state=C.SEED).fit_predict(Z)
            rows.append(dict(configurazione=name, k=k, silhouette=silhouette_score(Z, lab),
                             ARI_vs_profili_latenti=ari(truth, lab),
                             cluster_piu_grande=int(np.bincount(lab).max())))
    return pd.DataFrame(rows)


CAMPAIGN_ASSUMPTIONS, MASS_RATE, MASS_UPLIFT, CONTACT_COST = (
    C.CAMPAIGN_ASSUMPTIONS, C.MASS_RATE, C.MASS_UPLIFT, C.CONTACT_COST)


def campaign_simulation(prof, assume=CAMPAIGN_ASSUMPTIONS, mass_rate=MASS_RATE, mass_uplift=MASS_UPLIFT,
                        cost=CONTACT_COST, rate_factor=1.0):
    """Campagna di massa (A) contro campagna mirata per segmento (B)
    rate_factor scala i tassi di risposta di B (1.0 = ipotesi dichiarate): serve all'analisi di sensibilità"""
    n = int(prof["clienti"].sum())
    a_rev, a_cost = n * mass_rate * mass_uplift, n * cost
    b_rev = b_cost = 0.0
    b_contacts = 0
    for _, r in prof.iterrows():
        rate, inc, go = assume.get(r["segmento"], (mass_rate, mass_uplift, True))
        if go:
            b_rev += r["clienti"] * rate * rate_factor * inc
            b_cost += r["clienti"] * cost
            b_contacts += int(r["clienti"])
    # tasso di risposta di A per cui le due campagne si equivalgono (a parità di incremento per rispondente)
    breakeven_mass_rate = ((b_rev - b_cost) + a_cost) / (n * mass_uplift)
    return dict(A_incremento=a_rev, A_costo=a_cost, A_netto=a_rev - a_cost, A_contatti=n,
                B_incremento=b_rev, B_costo=b_cost, B_netto=b_rev - b_cost, B_contatti=b_contacts,
                rapporto_netto_B_su_A=(b_rev - b_cost) / (a_rev - a_cost) if (a_rev - a_cost) != 0 else float("nan"),
                tasso_A_di_pareggio=breakeven_mass_rate)


def campaign_sensitivity(prof, rate_factors=(0.5, 0.75, 1.0), costs=(0.20, 0.40, 0.80)):
    """Risultato netto di B e rapporto B/A al variare dei tassi di risposta di B e del costo per contatto"""
    rows = []
    for f in rate_factors:
        for c in costs:
            r = campaign_simulation(prof, cost=c, rate_factor=f)
            rows.append(dict(fattore_tassi_B=f, costo_contatto=c, A_netto=round(r["A_netto"], 1),
                             B_netto=round(r["B_netto"], 1), rapporto_B_su_A=round(r["rapporto_netto_B_su_A"], 2)))
    return pd.DataFrame(rows)


def trial_scenario(out, base_conv=C.TRIAL_BASE_CONV, uplift_pp=C.TRIAL_UPLIFT_PP):
    """Conversione dalla prova al piano Plus: valore delle conversioni attese e di quelle aggiuntive
    base_conv è il tasso di conversione ipotizzato oggi; uplift_pp i punti percentuali in più attesi
    dall'onboarding mirato"""
    n_trial = int((out["piano"] == "Trial").sum())
    plus = out.loc[out["piano"] == "Plus", "importo_totale_90g"].mean()
    trial = out.loc[out["piano"] == "Trial", "importo_totale_90g"].mean()
    base = n_trial * base_conv
    extra = n_trial * uplift_pp
    return dict(n_trial=n_trial, spesa_plus=plus, spesa_trial=trial,
                tasso_conversione_base=base_conv, conversioni_attese_oggi=base, valore_incrementale_oggi=base * (plus - trial),
                conversioni_aggiuntive=extra, valore_trimestrale=extra * (plus - trial))


def retrain_scenarios(df, seg, k):
    """Effetto del riaddestramento dopo l'arrivo di nuovi clienti simulati (scenari A-E)"""
    batch = G.generate(n=6000, seed=11)
    batch["customer_id"] = [f"N{i + 1:04d}" for i in range(len(batch))]
    tenure = (pd.Timestamp(C.REFERENCE_DATE) - pd.to_datetime(batch["data_iscrizione"])).dt.days
    scen = {"A: +40 clienti (stessa distribuzione)": batch.head(40),
            "B: +100 clienti (stessa distribuzione)": batch.head(100),
            "C: campagna di acquisizione (iscritti da < 30 giorni)": batch[tenure <= 30].head(300),
            "D: deriva verso l'inattività (nessuna richiesta recente)": batch[batch["richieste_90g"] == 0].head(150),
            "E: afflusso di clienti poco attivi (1-2 richieste, iscritti da > 60 giorni)":
                batch[batch["richieste_90g"].between(1, 2) & (tenure > 60)].head(400)}
    rows = []
    for name, new in scen.items():
        raw = pd.concat([S.load_data(), new], ignore_index=True)
        raw["data_iscrizione"] = pd.to_datetime(raw["data_iscrizione"])
        df_all, _ = S.clean_data(raw)
        r = N.retrain_comparison(df_all, seg, "kmeans", k)
        rows.append(dict(scenario=name, clienti_aggiunti=r["clienti_aggiunti"],
                         silhouette_prima=round(r["silhouette_prima"], 3), silhouette_dopo=round(r["silhouette_dopo"], 3),
                         ari=round(r["ari"], 3), quota_cambiata_pct=round(r["quota_cambiata_pct"], 1)))
    return pd.DataFrame(rows)


def stress_test(sizes=(500, 2000, 10000, 50000, 100000)):
    rows = []
    for n in sizes:
        d = G.generate(n=n, seed=1, missing_rate=0.03)
        d["data_iscrizione"] = pd.to_datetime(d["data_iscrizione"])
        t0 = time.perf_counter()
        d, _ = S.clean_data(d)
        pre = S.Preprocessor().fit(d)
        X = pre.transform(d)
        t1 = time.perf_counter()
        scan = S.scan_k(X) if n <= 10000 else S.scan_k(X, k_range=range(3, 6))
        t2 = time.perf_counter()
        model, lab = S.fit_labels(X, "kmeans", 5)
        t3 = time.perf_counter()
        sil = silhouette_score(X, lab, sample_size=min(n, 5000), random_state=0)
        rows.append(dict(n_clienti=n, prep_s=t1 - t0, scan_k_s=t2 - t1, fit_k5_s=t3 - t2, silhouette_campione=sil))
    return pd.DataFrame(rows)


def main():
    C.RESULTS_DIR.mkdir(exist_ok=True)
    df, report = S.clean_data(S.load_data())
    truth = latent_profiles(df)          # allineato per customer_id non per posizione
    pre = S.Preprocessor().fit(df)
    X = pre.transform(df)
    scan = S.scan_k(X)
    scan.to_csv(C.RESULTS_DIR / "valutazione_k.csv", index=False, float_format="%.10g")
    k = S.choose_k(scan)
    M = {"pulizia": report, "k_scelto": k}

    # confronto di configurazioni (k = 3, 4, 5) e di algoritmi (a k scelto)
    comp = []
    for algo in ("kmeans", "gmm", "agglomerative"):
        for kk in ((3, 4, 5) if algo == "kmeans" else (k,)):
            _, lab = S.fit_labels(X, algo, kk)
            comp.append(dict(algoritmo=algo, k=kk, silhouette=silhouette_score(X, lab),
                             ARI_vs_profili_latenti=ari(truth, lab)))
    pd.DataFrame(comp).to_csv(C.RESULTS_DIR / "confronto_configurazioni.csv", index=False)

    ablation(df, truth).to_csv(C.RESULTS_DIR / "ablazione_variabili.csv", index=False)
    seg = S.run_segmentation(df, k=k)
    out = S.assign_dataframe(df, seg)
    S.for_export(out).to_csv(C.RESULTS_DIR / "clienti_segmentati.csv", index=False)
    seg.profile.round(2).to_csv(C.RESULTS_DIR / "profili_segmenti.csv")
    M["profilo"] = seg.profile.round(2).reset_index().to_dict(orient="records")
    M["silhouette"] = float(silhouette_score(X, seg.labels))
    M["ARI_vs_latenti"] = float(ari(truth, seg.labels))
    M["stabilita_seed_mean_min"] = stability(X, k)                                   # configurazione adottata (n_init=20)
    M["stabilita_seed_n_init_1_random"] = stability(X, k, n_init=1, init="random")   # senza k-means++ nè ripetizioni
    M["stabilita_bootstrap_mean_std"] = bootstrap_stability(df, k)
    M["campagna"] = campaign_simulation(seg.profile)
    sens = campaign_sensitivity(seg.profile)
    sens.to_csv(C.RESULTS_DIR / "sensibilita_campagna.csv", index=False)
    M["campagna_sensibilita"] = sens.to_dict(orient="records")
    M["trial"] = trial_scenario(out)
    M["piani"] = out.groupby("piano")["importo_totale_90g"].agg(["count", "sum", "mean"]).round(1).reset_index().to_dict(orient="records")
    M["pareto_ricavi"] = {r["segmento"]: round(float(r["quota_ricavi_%"]), 1) for _, r in seg.profile.iterrows()}
    st = stress_test()
    st.to_csv(C.RESULTS_DIR / "stress_test.csv", index=False)
    M["stress"] = st.round(3).to_dict(orient="records")

    rt = retrain_scenarios(df, seg, k)
    rt.to_csv(C.RESULTS_DIR / "riaddestramento_scenari.csv", index=False)
    M["riaddestramento"] = rt.to_dict(orient="records")

    # stabilità rispetto al campionamento per k = 3, 4, 5 (a supporto della scelta di k)
    rows = []
    for kk in C.K_ALLOWED:
        m_, s_ = bootstrap_stability(df, kk)
        rows.append(dict(k=kk, ari_bootstrap_medio=round(m_, 3), ari_bootstrap_dev_std=round(s_, 3)))
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "stabilita_per_k.csv", index=False)
    M["stabilita_bootstrap_per_k"] = rows

    # riproducibilità -> due esecuzioni indipendenti producono la stessa partizione
    seg2 = S.run_segmentation(df, k=k)
    M["riproducibile"] = bool((seg2.labels == seg.labels).all())

    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(M, indent=2, ensure_ascii=False, default=float))
    print(json.dumps({k_: v for k_, v in M.items() if k_ not in ("profilo",)}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
