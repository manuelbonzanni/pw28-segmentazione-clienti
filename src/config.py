"""Parametri del progetto tutti in un unico file"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SEED = 42                      # seme globale -> dati e K-Means sono riproducibili
N_CUSTOMERS = 480              # range richiesto dalla traccia
REFERENCE_DATE = "2026-09-30"  # data di "fotografia" del database clienti
WINDOW_DAYS = 90               # finestra dell'"ultimo periodo" (richieste e importi)

ZONES = ["Nord-Ovest", "Nord-Est", "Centro", "Sud", "Isole"]
CATEGORIES = ["Pulizie", "Riparazioni", "Giardinaggio", "Traslochi", "Assistenza IT"]
PLANS = ["Pay-per-use", "Plus", "Trial"]

# Variabili RFM (Recency, Frequency, Monetary) + anzianità (giorni dall'iscrizione, derivata da data_iscrizione)
RFM_FEATURES = ["richieste_90g", "importo_totale_90g", "giorni_ultima_richiesta"]
FEATURES = RFM_FEATURES + ["anzianita_giorni"]   # variabili usate dal clustering
# Variabili ad asimmetria forte -> si applica log(1+x) prima della standardizzazione
LOG_FEATURES = ["richieste_90g", "importo_totale_90g"]

K_RANGE = range(2, 9)          # intervallo esplorato per gomito/silhouette
K_ALLOWED = (3, 4, 5)          # gruppi richiesti dalla traccia
N_INIT = 20                    # inizializzazioni di K-Means (k-means++)

# Soglie usate da name_segments per dare un nome ai gruppi 
NEW_TENURE_RATIO = 0.6         # un gruppo è "nuovo" se la sua anzianità media < 60% della mediana fra i gruppi
REGULAR_MIN_REQUESTS = 3       # "regolari" se in media almeno 3 richieste in 90 giorni (una al mese)

# Soglie di assegnazione e di visualizzazione
ASSIGN_MARGIN = 0.8            # assegnazione "poco netta" se dist(segmento assegnato) / dist(secondo segmento) > 0,8
MIN_VIEW = 10                  # sotto questo numero di clienti la dashboard non mostra grafici e riepiloghi (non avrebbero senso)

# ----------------------------------------------------------------------------- parametri del generatore sintetico
# nome: (quota, lambda richieste, media ticket, dev.std ticket, range recency, range tenure in giorni,
#        probabilità piano Plus, probabilità piano Trial)
ARCHETYPES = {
    "fedeli":      dict(share=0.17, lam=9.0, ticket=58, sd=0.30, rec=(1, 14),   ten=(120, 270), plus=0.70, trial=0.00),
    "regolari":    dict(share=0.30, lam=3.6, ticket=42, sd=0.30, rec=(3, 40),   ten=(90, 270),  plus=0.30, trial=0.00),
    "occasionali": dict(share=0.23, lam=1.3, ticket=34, sd=0.35, rec=(25, 90),  ten=(100, 270), plus=0.05, trial=0.00),
    "nuovi":       dict(share=0.14, lam=1.8, ticket=37, sd=0.35, rec=(0, 12),   ten=(2, 30),    plus=0.00, trial=0.75),
    "inattivi":    dict(share=0.16, lam=0.0, ticket=36, sd=0.35, rec=(91, 240), ten=(130, 270), plus=0.02, trial=0.00),
}
# preferenze di categoria per profilo (probabilità sulle 5 categorie di config.CATEGORIES)
CAT_PREF = {
    "fedeli":      [0.30, 0.25, 0.10, 0.05, 0.30],
    "regolari":    [0.35, 0.20, 0.20, 0.05, 0.20],
    "occasionali": [0.15, 0.35, 0.15, 0.25, 0.10],
    "nuovi":       [0.40, 0.15, 0.10, 0.10, 0.25],
    "inattivi":    [0.15, 0.30, 0.20, 0.30, 0.05],
}
ZONE_P = [0.30, 0.22, 0.22, 0.18, 0.08]

# ----------------------------------------------------------------------------- ipotesi della simulazione (relazione, sez. Indicatori)
# segmento -> (tasso di risposta, incremento EUR per rispondente, contattare?)
CAMPAIGN_ASSUMPTIONS = {
    "Clienti fedeli ad alto valore": (0.10, 30, False),
    "Clienti occasionali": (0.08, 24, True),
    "Nuovi clienti in fase di prova": (0.18, 22, True),
    "Inattivi / a rischio abbandono": (0.06, 40, True),
}
MASS_RATE, MASS_UPLIFT, CONTACT_COST = 0.04, 25, 0.40   # campagna generica -> tasso, incremento EUR, costo per contatto
TRIAL_BASE_CONV, TRIAL_UPLIFT_PP = 0.20, 0.05           # conversione prova -> Plus oggi, punti percentuali in più attesi
