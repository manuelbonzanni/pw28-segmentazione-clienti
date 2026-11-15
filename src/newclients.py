"""Nuovi clienti inseriti dalla dashboard e riaddestramento controllato"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

import config as C
import segmentation as S

NEW_PATH = C.DATA_DIR / "clienti_nuovi.csv"
STATE_PATH = C.DATA_DIR / "modello_attivo.json"
BASE_COLUMNS = ["customer_id", "zona", "data_iscrizione", "richieste_90g", "importo_totale_90g",
                "giorni_ultima_richiesta", "categoria_preferita", "piano"]
EXTRA_COLUMNS = ["segmento_assegnato", "inserito_il"]
ID_PREFIX = "N"  # i clienti nuovi inseriti dalla dashboard iniziano con N, quelli creati dal generatore con C


# ----------------------------------------------------------------------------- validazione
def validate_new_client(rec):
    """Controlla un record: con errori il salvataggio è bloccato, gli avvisi no"""
    n, amount, days = rec["richieste_90g"], rec["importo_totale_90g"], rec["giorni_ultima_richiesta"]
    tenure = (pd.Timestamp(C.REFERENCE_DATE) - pd.Timestamp(rec["data_iscrizione"])).days
    errors, warnings = [], []
    if tenure < 0:
        errors.append(f"La data di iscrizione è successiva alla data di riferimento del database ({C.REFERENCE_DATE}).")
    if n == 0 and amount > 0:
        errors.append("Importo maggiore di zero con 0 richieste negli ultimi 90 giorni: i due valori sono incoerenti.")
    if n > 0 and amount <= 0:
        errors.append("Ci sono richieste negli ultimi 90 giorni ma l'importo è zero: inserisci l'importo speso.")
    if n == 0 and days <= C.WINDOW_DAYS:
        errors.append(f"Con 0 richieste negli ultimi {C.WINDOW_DAYS} giorni l'ultima richiesta deve risalire a oltre "
                      f"{C.WINDOW_DAYS} giorni fa.")
    if n > 0 and days > C.WINDOW_DAYS:
        errors.append(f"Ci sono richieste negli ultimi {C.WINDOW_DAYS} giorni, quindi l'ultima richiesta non può "
                      f"risalire a oltre {C.WINDOW_DAYS} giorni fa.")
    if tenure >= 0 and days > tenure:
        errors.append(f"L'ultima richiesta ({int(days)} giorni fa) è precedente all'iscrizione ({tenure} giorni fa).")
    if n <= 1:
        warnings.append("Con 0 o 1 richieste negli ultimi 90 giorni la regolarità del cliente non è valutabile: "
                        "il segmento assegnato è solo indicativo.")
    return errors, warnings


# ----------------------------------------------------------------------------- archivio nuovi clienti
def load_new(path=None):
    path = path or NEW_PATH
    if not path.exists():
        return pd.DataFrame(columns=BASE_COLUMNS + EXTRA_COLUMNS)
    return pd.read_csv(path, parse_dates=["data_iscrizione"])


def next_id(existing):
    nums = [int(i[len(ID_PREFIX):]) for i in existing if str(i).startswith(ID_PREFIX) and str(i)[1:].isdigit()]
    return f"{ID_PREFIX}{(max(nums) if nums else 0) + 1:04d}"


def append_new(rec, segment, path=None):
    path = path or NEW_PATH
    cur = load_new(path)
    row = dict(rec)
    row["customer_id"] = next_id(cur["customer_id"])
    row["data_iscrizione"] = pd.Timestamp(rec["data_iscrizione"]).date().isoformat()
    row["segmento_assegnato"] = segment
    row["inserito_il"] = datetime.now().isoformat(timespec="seconds")
    out = pd.concat([cur, pd.DataFrame([row])[BASE_COLUMNS + EXTRA_COLUMNS]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return row


# ----------------------------------------------------------------------------- modello attivo
def get_active_count(path=None):
    path = path or STATE_PATH
    try:
        return int(json.loads(path.read_text())["clienti_nuovi_inclusi"])
    except (OSError, ValueError, KeyError):
        return 0


def set_active_count(n, path=None):
    path = path or STATE_PATH
    path.write_text(json.dumps({"clienti_nuovi_inclusi": int(n),
                                "aggiornato_il": datetime.now().isoformat(timespec="seconds")}, indent=2))


def training_data(n_new, base=None, new_path=None):
    if base is None:
        base = S.load_data()
    new = load_new(new_path).head(n_new)[BASE_COLUMNS]
    df = pd.concat([base[BASE_COLUMNS], new], ignore_index=True)
    df["data_iscrizione"] = pd.to_datetime(df["data_iscrizione"])
    return S.clean_data(df)


# ----------------------------------------------------------------------------- riaddestramento controllato
def retrain_comparison(df_all, seg_old, algo, k):
    """Confronta il modello in uso con uno riaddestrato su df_all senza modificare nulla"""
    n_old = len(seg_old.labels)
    if not np.array_equal(df_all["customer_id"].to_numpy()[:n_old], seg_old.customer_ids):
        raise ValueError("I primi clienti di df_all non corrispondono a quelli del modello in uso: confronto non valido")
    seg_new = S.run_segmentation(df_all, k=k, algo=algo)
    old_lab, new_lab = seg_old.labels, seg_new.labels[:n_old]
    old_names = pd.Series(old_lab).map(seg_old.names)
    new_names = pd.Series(new_lab).map(seg_new.names)
    prof_old = seg_old.profile.set_index("segmento")
    prof_new = seg_new.profile.set_index("segmento")
    table = pd.DataFrame({
        "clienti_prima": prof_old["clienti"], "clienti_dopo": prof_new["clienti"],
        "quota_ricavi_prima_%": prof_old["quota_ricavi_%"], "quota_ricavi_dopo_%": prof_new["quota_ricavi_%"],
    }).fillna(0).round(1)
    table[["clienti_prima", "clienti_dopo"]] = table[["clienti_prima", "clienti_dopo"]].astype(int)
    return dict(
        seg_new=seg_new,
        clienti_aggiunti=len(df_all) - n_old,
        silhouette_prima=float(silhouette_score(seg_old.X, old_lab)),
        silhouette_dopo=float(silhouette_score(seg_new.X, seg_new.labels)),
        ari=float(adjusted_rand_score(old_lab, new_lab)),
        quota_cambiata_pct=float(100 * (old_names.values != new_names.values).mean()),
        tabella=table,
    )
