"""Pipeline di segmentazione: pulizia, normalizzazione, K-Means, valutazione e profili"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import config as C


# ----------------------------------------------------------------------------- 1. dati
def add_tenure(df):
    """Calcola l'anzianità dalla data di iscrizione se manca"""
    if "anzianita_giorni" not in df.columns and "data_iscrizione" in df.columns:
        df = df.copy()
        df["anzianita_giorni"] = (pd.Timestamp(C.REFERENCE_DATE) - pd.to_datetime(df["data_iscrizione"])).dt.days
    return df


def load_data(path=None):
    df = pd.read_csv(path or C.DATA_DIR / "clienti.csv", parse_dates=["data_iscrizione"])
    return df


def clean_data(df):
    """Rimuove duplicati, valori impossibili e mancanti e restituisce dati e report"""
    report = {"righe_iniziali": len(df)}
    df = df.drop_duplicates(subset="customer_id").copy()
    report["duplicati_rimossi"] = report["righe_iniziali"] - len(df)

    # valori impossibili trattati come mancanti
    for col in ["richieste_90g", "importo_totale_90g", "giorni_ultima_richiesta"]:
        bad = df[col] < 0
        report[f"negativi_{col}"] = int(bad.sum())
        df.loc[bad, col] = np.nan

    report["mancanti_richieste"] = int(df["richieste_90g"].isna().sum())
    report["mancanti_importo"] = int(df["importo_totale_90g"].isna().sum())
    report["mancanti_giorni"] = int(df["giorni_ultima_richiesta"].isna().sum())

    # imputazione delle richieste mancanti
    miss_n = df["richieste_90g"].isna()
    if miss_n.any():
        fuori_finestra = miss_n & (df["giorni_ultima_richiesta"] > C.WINDOW_DAYS)
        df.loc[fuori_finestra, "richieste_90g"] = 0
        med_n = df.loc[df["richieste_90g"] > 0, "richieste_90g"].median()
        df.loc[miss_n & ~fuori_finestra, "richieste_90g"] = med_n

    # imputazione -> importo = n richieste * mediana dell'importo medio x richiesta
    # giorni dall'ultima richiesta = mediana 
    active = df["richieste_90g"] > 0
    median_ticket = (df.loc[active, "importo_totale_90g"] / df.loc[active, "richieste_90g"]).median()
    miss = df["importo_totale_90g"].isna()
    df.loc[miss, "importo_totale_90g"] = (df.loc[miss, "richieste_90g"] * median_ticket).round(2)
    for flag in (True, False):
        m = (active == flag) & df["giorni_ultima_richiesta"].isna()
        df.loc[m, "giorni_ultima_richiesta"] = df.loc[active == flag, "giorni_ultima_richiesta"].median()
    report["righe_finali"] = len(df)
    return add_tenure(df).reset_index(drop=True), report


# ----------------------------------------------------------------------------- 2. feature
@dataclass
class Preprocessor:
    """log(1+x) sulle variabili asimmetriche e poi z-score"""
    scaler: StandardScaler = None
    features: tuple = tuple(C.FEATURES)

    def _log(self, X):
        X = add_tenure(X)[list(self.features)].astype(float).copy()
        for c in C.LOG_FEATURES:
            X[c] = np.log1p(X[c])
        return X

    def fit(self, df):
        self.scaler = StandardScaler().fit(self._log(df))
        return self

    def transform(self, df):
        return self.scaler.transform(self._log(df))


# ----------------------------------------------------------------------------- 3. modelli
def make_model(algo="kmeans", k=4, seed=C.SEED):
    if algo == "kmeans":
        return KMeans(n_clusters=k, init="k-means++", n_init=C.N_INIT, random_state=seed)
    if algo == "gmm":
        return GaussianMixture(n_components=k, n_init=5, random_state=seed)
    if algo == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage="ward")
    raise ValueError(f"algoritmo sconosciuto: {algo}")


def fit_labels(X, algo="kmeans", k=4, seed=C.SEED):
    model = make_model(algo, k, seed)
    labels = model.fit_predict(X)
    return model, labels


def scan_k(X, k_range=C.K_RANGE, seed=C.SEED):
    """Indici di qualità per ogni k (inerzia, silhouette, Davies-Bouldin, Calinski-Harabasz)"""
    rows = []
    for k in k_range:
        model, labels = fit_labels(X, "kmeans", k, seed)
        rows.append(dict(k=k, inertia=model.inertia_,
                         # su dataset grandi la silhouette (O(n^2)) è calcolata su un campione casuale
                         silhouette=silhouette_score(X, labels, sample_size=min(len(X), 5000), random_state=seed),
                         davies_bouldin=davies_bouldin_score(X, labels),
                         calinski_harabasz=calinski_harabasz_score(X, labels)))
    return pd.DataFrame(rows)


def choose_k(scan, allowed=C.K_ALLOWED):
    """Tra i k ammessi sceglie quello con la silhouette massima"""
    sub = scan[scan["k"].isin(allowed)]
    return int(sub.loc[sub["silhouette"].idxmax(), "k"])


# ----------------------------------------------------------------------------- 4. profili
def name_segments(profile):
    """Da un nome di marketing a ogni gruppo partendo dai centroidi"""
    names, left = {}, set(profile.index)
    def take(label, idx):
        names[idx] = label
        left.discard(idx)
    if profile.empty:
        return names
    take("Inattivi / a rischio abbandono", profile["giorni_ultima_richiesta"].idxmax())
    if left:
        cand = list(left)
        take("Clienti fedeli ad alto valore", profile.loc[cand, "importo_totale_90g"].idxmax())
    if left:
        # il più recente tra i rimanenti con tenure minima è "nuovo"
        cand = list(left)
        key = profile.loc[cand, "anzianita_giorni"].idxmin()
        if profile.loc[key, "anzianita_giorni"] < C.NEW_TENURE_RATIO * profile["anzianita_giorni"].median():
            take("Nuovi clienti in fase di prova", key)
    rest = sorted(left, key=lambda j: -profile.loc[j, "richieste_90g"])
    if len(rest) == 1:
        # un solo gruppo rimasto -> "regolare" se in media almeno una richiesta al mese altrimenti "occasionale"
        names[rest[0]] = ("Clienti regolari" if profile.loc[rest[0], "richieste_90g"] >= C.REGULAR_MIN_REQUESTS
                          else "Clienti occasionali")
    else:
        for i in rest:
            if "Clienti regolari" not in names.values():
                names[i] = "Clienti regolari"
            elif "Clienti occasionali" not in names.values():
                names[i] = "Clienti occasionali"
            else:
                names[i] = f"Gruppo {i}"
    return names


def profile_segments(df, labels, names=None):
    """Profilo di ogni gruppo, con names usa le etichette già assegnate"""
    d = add_tenure(df).copy()
    d["cluster"] = labels
    prof = d.groupby("cluster").agg(
        clienti=("customer_id", "count"),
        richieste_90g=("richieste_90g", "mean"),
        importo_totale_90g=("importo_totale_90g", "mean"),
        giorni_ultima_richiesta=("giorni_ultima_richiesta", "mean"),
        anzianita_giorni=("anzianita_giorni", "mean"),
        ricavo_totale=("importo_totale_90g", "sum"),
    )
    prof["quota_clienti_%"] = 100 * prof["clienti"] / prof["clienti"].sum()
    prof["quota_ricavi_%"] = 100 * prof["ricavo_totale"] / prof["ricavo_totale"].sum()
    prof["categoria_prevalente"] = d.groupby("cluster")["categoria_preferita"].agg(lambda s: s.value_counts().idxmax())
    prof["quota_plus_%"] = 100 * d.groupby("cluster")["piano"].apply(lambda s: (s == "Plus").mean())
    prof["quota_trial_%"] = 100 * d.groupby("cluster")["piano"].apply(lambda s: (s == "Trial").mean())
    if names is not None:
        prof["segmento"] = pd.Series({i: names.get(i, f"Gruppo {i}") for i in prof.index})
    else:
        prof["segmento"] = pd.Series(name_segments(prof))
    return prof


# ----------------------------------------------------------------------------- 5. modello addestrato
@dataclass
class SegmentationModel:
    pre: Preprocessor
    model: object
    k: int
    algo: str
    names: dict
    profile: pd.DataFrame
    labels: np.ndarray
    X: np.ndarray

    def predict(self, new_df):
        """Assegna un segmento ai nuovi clienti con la stessa trasformazione dell'addestramento"""
        Xn = self.pre.transform(new_df)
        if hasattr(self.model, "predict"):
            lab = self.model.predict(Xn)
        else:  # AgglomerativeClustering non ha predict -> si usa il centroide più vicino
            cent = np.vstack([self.X[self.labels == c].mean(axis=0) for c in range(self.k)])
            lab = np.argmin(((Xn[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        return lab, [self.names[i] for i in lab]


    def explain(self, new_df, margin_uncertain=C.ASSIGN_MARGIN):
        """Segmento assegnato a un cliente, secondo segmento più vicino se l'assegnazione è poco netta"""
        lab, _ = self.predict(new_df)
        a = int(lab[0])
        Xn = self.pre.transform(new_df)[0]
        cent = {c: self.X[self.labels == c].mean(axis=0) for c in range(self.k) if (self.labels == c).any()}
        dist = {c: float(np.linalg.norm(Xn - m)) for c, m in cent.items()}
        others = sorted((c for c in dist if c != a), key=dist.get)
        if not others:
            return dict(segmento=self.names[a], secondo=None, rapporto=None, incerto=False)
        b = others[0]
        ratio = dist[a] / dist[b] if dist[b] > 0 else 1.0
        return dict(segmento=self.names[a], secondo=self.names[b], rapporto=ratio, incerto=ratio > margin_uncertain)


def run_segmentation(df, k=None, algo="kmeans", seed=C.SEED):
    pre = Preprocessor().fit(df)
    X = pre.transform(df)
    if k is None:
        k = choose_k(scan_k(X, seed=seed))
    model, labels = fit_labels(X, algo, k, seed)
    prof = profile_segments(df, labels)
    seg = SegmentationModel(pre, model, k, algo, prof["segmento"].to_dict(), prof, labels, X)
    seg.customer_ids = df["customer_id"].to_numpy()  # ordine delle righe addestrate x il confronto con il riaddestramento
    return seg


def for_export(df):
    """Copia per l'esportazione CSV: le colonne di conteggio intere (richieste, giorni) senza '.0'
    Non modifica i dati del modello: l'imputazione resta invariata"""
    out = df.copy()
    for col in ["richieste_90g", "giorni_ultima_richiesta"]:
        if col not in out.columns or len(out) == 0:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        if not vals.notna().all():
            continue
        arr = vals.to_numpy(dtype=float)
        if np.allclose(arr, np.round(arr)):
            out[col] = vals.round().astype("Int64")
    return out


def assign_dataframe(df, seg):
    out = df.copy()
    out["cluster"] = seg.labels
    out["segmento"] = out["cluster"].map(seg.names)
    # anzianità (giorni dall'iscrizione)-> serve come asse selezionabile nella dashboard
    out["anzianita_giorni"] = (pd.Timestamp(C.REFERENCE_DATE) - out["data_iscrizione"]).dt.days
    return out
