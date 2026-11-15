# CasaPronta - Segmentazione automatica della clientela

Prototipo in Python che genera dati sintetici sulla clientela di una startup fittizia di servizi domestici on demand, individua gruppi di clienti con **K-Means** (Machine Learning non supervisionato), valuta la qualità dei gruppi (gomito, silhouette, Davies-Bouldin, Calinski-Harabasz, stabilità) 
e li presenta in una **dashboard** interattiva.

## Requisiti
Python 3.10+ (testato con 3.13; il workflow GitHub Actions in `.github/workflows/github_workflows_test.yml` esegue i test con 3.13 e verifica che
`generate_data.py` riproduca esattamente `data/`). Versioni esatte delle librerie in `requirements.txt`; vincoli minimi compatibili in `requirements-min.txt`.

## Esecuzione

```bash
python -m venv .venv
.venv\Scripts\activate            
pip install -r requirements.txt
python src/generate_data.py       # 1) genera data/clienti.csv (480 clienti, seed 42)
python src/evaluate.py            # 2) calcola segmenti e metriche -> results/
streamlit run src/app.py          # 3) apre la dashboard su http://localhost:8501
python -m unittest discover -s tests -v   # 4) test automatici (opzionale, il test della dashboard è saltato se Streamlit non è installato)
```

I file `data/clienti.csv` e `results/*` sono già inclusi nel repository: la dashboard parte anche senza rieseguire 1) e 2).
In `results/metrics.json` e `results/stress_test.csv` i tempi di esecuzione dipendono dalla macchina e cambiano a ogni run, tutti gli altri risultati sono deterministici.

## Struttura

| Percorso | Contenuto |
|---|---|
| `src/config.py` | Parametri centrali: seme, numero clienti, variabili usate, intervallo di k, soglie per i nomi dei segmenti |
| `src/generate_data.py` | Generatore dei dati sintetici (5 profili comportamentali latenti, valori mancanti simulati) |
| `src/segmentation.py` | Pulizia, log + standardizzazione, K-Means / GMM / Agglomerativo, scelta di k, profili e nomi dei segmenti, predizione su nuovi clienti |
| `src/newclients.py` | Validazione e salvataggio dei nuovi clienti inseriti dalla dashboard, riaddestramento controllato con confronto prima/dopo |
| `src/plots.py` | Grafici |
| `src/evaluate.py` | Metriche, confronto configurazioni, stabilità (anche senza k-means++), simulazione di impatto con sensibilità, stress test |
| `src/app.py` | Dashboard Streamlit |
| `tests/` | Test unitari/funzionali e smoke test della dashboard |
| `data/clienti.csv` | Dataset sintetico una riga per cliente |
| `data/profilo_latente_verita.csv` | Profilo "vero" usato dal generatore, **solo per validare** il clustering |
| `data/clienti_nuovi.csv`, `data/modello_attivo.json` | Generati dall'uso della dashboard (clienti inseriti; quanti sono inclusi nel modello). **Non versionati** |
| `results/clienti_segmentati.csv` | Risultato del raggruppamento (cliente -> cluster -> segmento) |
| `results/profili_segmenti.csv`, `valutazione_k.csv`, `confronto_configurazioni.csv`, `ablazione_variabili.csv`, `stabilita_per_k.csv`, `riaddestramento_scenari.csv`, `sensibilita_campagna.csv`, `stress_test.csv`, `metrics.json` | Tabelle di risultato e metriche |

## Dati (data/clienti.csv)
`customer_id, zona, data_iscrizione, richieste_90g, importo_totale_90g, giorni_ultima_richiesta, categoria_preferita, piano`
(`piano` = Pay-per-use / Plus / Trial). Data di riferimento: 2026-09-30. Circa il 3% di importi e giorni sono mancanti di proposito e vengono imputati in `clean_data` (richieste mancanti: 0 se l'ultima richiesta è oltre 90 giorni, altrimenti mediana degli attivi; importo: richieste x mediana del ticket; giorni: mediana separata fra attivi e inattivi).

## Riproducibilità
Tutto dipende da `SEED = 42` in `src/config.py`. Due esecuzioni producono lo stesso CSV e la stessa partizione
(verificato da `tests/test_pipeline.py` e da `metrics.json -> riproducibile`).

## Dashboard
Barra laterale: algoritmo, numero di gruppi (3-5), filtri per zona, categoria, piano e periodo di iscrizione.
Schede: *Panoramica* (scatter con assi a scelta, numerosità e ricavi), *Profili dei segmenti* (heatmap, tabella,
azioni di marketing), *Scelta di k* (gomito, silhouette), *Risultati per cliente* (tabella ricercabile e scaricabile in CSV),
*Nuovo cliente* (modulo -> riepilogo e segmento assegnato -> conferma e salvataggio).

**Nuovi clienti e aggiornamento del modello.** I clienti confermati nella scheda *Nuovo cliente* sono accodati a `data/clienti_nuovi.csv`; 
`data/clienti.csv` non viene mai modificato. Il modello in uso cambia solo con il pulsante *Mostra l'effetto del riaddestramento* seguito da *Adotta il nuovo modello* (confronto di silhouette, quota di clienti che cambiano segmento e ARI); *Torna al modello originale* annulla l'adozione. 
Per ripartire da zero basta cancellare i due file generati in `data/`.
