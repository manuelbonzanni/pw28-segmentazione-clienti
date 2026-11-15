"""Dashboard Streamlit per la segmentazione dei clienti"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score

import config as C
import newclients as N
import plots as P
import segmentation as S

st.set_page_config(page_title="CasaPronta - Segmentazione clienti", layout="wide")

ACTIONS = {
    "Clienti fedeli ad alto valore": "Programma fedeltà e accesso anticipato ai nuovi servizi; nessuno sconto generalizzato (margine).",
    "Clienti regolari": "Proposta di passaggio all'abbonamento Plus con prova del secondo servizio (cross-sell).",
    "Clienti occasionali": "Promemoria stagionali sulla categoria preferita, bundle a pacchetto per aumentare la frequenza e prova del piano Plus a chi ha già almeno tre richieste.",
    "Nuovi clienti in fase di prova": "Onboarding guidato e coupon sulla seconda richiesta prima della fine del trial.",
    "Inattivi / a rischio abbandono": "Campagna di riattivazione con offerta a tempo; dopo 2 tentativi senza risposta, sospendere le comunicazioni.",
}


def show_fig(fig):
    """Mostra la figura e la chiude x non lasciare grafici aperti in memoria"""
    st.pyplot(fig)
    plt.close(fig)


def tabella(data, **kw):
    """Tabella a tutta larghezza x compatibilità versioni vecchie e nuove di Streamlit"""
    try:
        st.dataframe(data, width="stretch", **kw)
    except Exception:
        st.dataframe(data, use_container_width=True, **kw)


@st.cache_data
def load(n_new):
    return N.training_data(n_new)


@st.cache_resource
def train(algo, k, n_new):
    """Modello addestrato e sua silhouette media: entrambi in cache così il cambio di un filtro non li ricalcola"""
    df, _ = load(n_new)
    seg = S.run_segmentation(df, k=k, algo=algo)
    return seg, float(silhouette_score(seg.X, seg.labels))


@st.cache_data
def scan(n_new):
    df, _ = load(n_new)
    X = S.Preprocessor().fit(df).transform(df)
    return S.scan_k(X)


n_new = N.get_active_count()   # nuovi clienti inclusi nel modello in uso (cambia solo con un riaddestramento)
df, report = load(n_new)
sc = scan(n_new)
k_auto = S.choose_k(sc)

# ------------------------------------------------------------------ sidebar: parametri e filtri
st.sidebar.header("Modello")
algo = st.sidebar.selectbox("Algoritmo", ["kmeans", "gmm", "agglomerative"],
                            format_func={"kmeans": "K-Means", "gmm": "Gaussian Mixture", "agglomerative": "Agglomerativo (Ward)"}.get)
k = st.sidebar.slider("Numero di gruppi (k)", 3, 5, k_auto)
seg, sil = train(algo, k, n_new)
data = S.assign_dataframe(df, seg)

def multi_filter(container, label, options):
    """Filtro a scelta multipla, x compatibilità versioni previsto pills se disponibili altrimenti multiselect"""
    if hasattr(container, "pills"):
        return container.pills(label, options, selection_mode="multi", default=list(options),
                               help="Clic su una voce per attivarla o disattivarla.")
    return container.multiselect(label, options, default=list(options))


st.sidebar.header("Filtri")
zone = multi_filter(st.sidebar, "Zona", C.ZONES)
cats = multi_filter(st.sidebar, "Categoria preferita", C.CATEGORIES)
plans = multi_filter(st.sidebar, "Piano", C.PLANS)
dmin, dmax = data["data_iscrizione"].min().date(), data["data_iscrizione"].max().date()
period = st.sidebar.date_input("Periodo di iscrizione", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(period, tuple) and len(period) == 2:
    d0, d1 = period
else:
    d0, d1 = dmin, dmax

view = data[data["zona"].isin(zone) & data["categoria_preferita"].isin(cats) & data["piano"].isin(plans)
            & data["data_iscrizione"].dt.date.between(d0, d1)]
cmap = P.color_map(data["segmento"])
MIN_VIEW = C.MIN_VIEW  # sotto questa soglia grafici e riepiloghi non sono significativi quindi si avvisa l'utente
too_few = len(view) < MIN_VIEW
NO_DATA_MSG = ("Con i filtri attuali ci sono troppo pochi clienti ({n}) per produrre risultati affidabili (minimo {m}). "
               "Allarga il periodo di iscrizione o seleziona più zone, categorie e piani nella barra laterale.")

# ------------------------------------------------------------------ intestazione e KPI
st.title("CasaPronta - Segmentazione automatica della clientela")
st.caption("Dati interamente sintetici. Il modello è addestrato su tutti i clienti; "
           "i filtri a sinistra cambiano solo la porzione visualizzata.")
if too_few:
    st.warning(NO_DATA_MSG.format(n=len(view), m=MIN_VIEW))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clienti nel filtro", f"{len(view)} / {len(data)}", help="Clienti che rispettano i filtri della barra laterale.")
c2.metric("Ricavi 90g nel filtro", f"{view['importo_totale_90g'].sum():,.0f} EUR")
c3.metric("Silhouette del modello", f"{sil:.3f}",
          help="Indice tra -1 e 1: quanto i gruppi sono ben separati. Sopra 0,5 = struttura ragionevole; 0,26-0,50 = debole.")
c4.metric("Segmenti", k)
piani_v = view["piano"].value_counts().reindex(["Pay-per-use", "Plus", "Trial"], fill_value=0)
st.caption("Piani nel filtro: " + " · ".join(f"{p} {n}" for p, n in piani_v.items()))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Panoramica", "Profili dei segmenti", "Scelta di k", "Risultati per cliente", "Nuovo cliente"])

# ------------------------------------------------------------------ tab 1: scatter
with tab1:
    if too_few:
        st.warning(NO_DATA_MSG.format(n=len(view), m=MIN_VIEW))
    else:
        a, b = st.columns([3, 2], gap="large")
        with a:
            c_x, c_y, c_h = st.columns(3)
            x = c_x.selectbox("Asse X", P.LABELS.keys(), index=2, format_func=P.LABELS.get)
            y = c_y.selectbox("Asse Y", P.LABELS.keys(), index=1, format_func=P.LABELS.get)
            hl = c_h.selectbox("Evidenzia segmento", ["Tutti"] + sorted(data["segmento"].unique()),
                               help="Colora un solo segmento e rende grigi gli altri: utile dove i punti si sovrappongono.")
            if x == y: # non significativo fare lo scatter con stessa variabile su entrambi gli assi, avviso l'utente
                st.info("Hai scelto la stessa variabile sui due assi: seleziona due variabili diverse per leggere il grafico.")
            show_fig(P.scatter_fig(view, x, y, cmap, highlight=None if hl == "Tutti" else hl))
            st.caption("Ogni punto è un cliente; colore e forma indicano il segmento. In sintesi - " + P.describe_scatter(view, x, y))
        with b:
            st.subheader("Numerosità e ricavi")
            prof_v = S.profile_segments(view, view["cluster"].values, names=seg.names)
            show_fig(P.size_revenue_fig(prof_v, cmap))
            st.caption("Barra grigia: quota di clienti. Barra colorata: quota dei ricavi del periodo.")
        st.subheader("Categoria preferita per zona")
        show_fig(P.zone_category_fig(view))
        st.caption("Ogni barra è una zona; i colori mostrano la quota di clienti per categoria preferita. "
                   "Il numero tra parentesi è la numerosità della zona nel filtro.")

# ------------------------------------------------------------------ tab 2: profili
with tab2:
    st.caption("I profili descrivono il modello e sono calcolati su tutti i clienti: i filtri della barra laterale "
               "non li modificano, così un segmento mantiene lo stesso significato qualunque sia la selezione.")
    prof = seg.profile.copy()
    cols = ["richieste_90g", "importo_totale_90g", "giorni_ultima_richiesta", "anzianita_giorni"]
    show_fig(P.profile_heatmap(prof, cols))
    show = prof[["segmento", "clienti", "quota_clienti_%", "quota_ricavi_%", *cols, "categoria_prevalente",
                 "quota_plus_%", "quota_trial_%"]].round(1)
    tabella(show)
    st.subheader("Interpretazione e azione di marketing suggerita")
    for _, r in prof.iterrows():
        st.markdown(f"**{r['segmento']}** ({int(r['clienti'])} clienti, {r['quota_ricavi_%']:.0f}% dei ricavi) - "
                    f"{ACTIONS.get(r['segmento'], 'Nessuna azione predefinita: ispezionare il profilo.')}")

# ------------------------------------------------------------------ tab 3: scelta di k
with tab3:
    show_fig(P.elbow_fig(sc, k_sel=k))
    tabella(sc.round(3))
    st.info(f"Regola adottata: il k (3-5) con la silhouette massima. Valore suggerito: k = {k_auto}.")
    show_fig(P.silhouette_plot(seg.X, seg.labels, seg.names, cmap))

# ------------------------------------------------------------------ tab 4: risultati consultabili
with tab4:
    q = st.text_input("Cerca per Customer_ID (es. C0012)")
    only = st.multiselect("Segmento", sorted(data["segmento"].unique()), default=sorted(data["segmento"].unique()))
    table = view[view["segmento"].isin(only)]
    if q:
        table = table[table["customer_id"].str.contains(q.strip(), case=False)]
    st.write(f"{len(table)} clienti")
    if table.empty:
        st.warning("Nessun cliente corrisponde ai filtri o alla ricerca: modifica i filtri della barra laterale, "
                   "i segmenti scelti o l'ID cercato.")
    tabella(table.drop(columns="cluster"), height=380)
    st.download_button("Scarica risultati (CSV)", S.for_export(table).to_csv(index=False).encode("utf-8"),
                       file_name="clienti_segmentati_filtrati.csv", mime="text/csv")

# ------------------------------------------------------------------ tab 5: nuovo cliente
FIELD_HELP = {
    "richieste": "Numero di richieste di servizio effettuate dal cliente negli ultimi 90 giorni.",
    "importo": "Somma spesa negli ultimi 90 giorni, in euro. Deve essere zero solo se le richieste sono zero.",
    "giorni": "Giorni trascorsi dalla richiesta più recente. Con una sola richiesta sono i giorni passati da quell'unico "
              "acquisto (0 = oggi). Con 0 richieste negli ultimi 90 giorni deve essere maggiore di 90.",
}
with tab5:
    st.write("Inserisci i dati di un cliente: il modello in uso gli assegna un segmento. Il cliente viene salvato "
             "solo dopo la tua conferma in un file separato (i dati originali non vengono toccati).") # x mantenere il modello rieseguibile senza modificare il dataset originale
    with st.form("nuovo_cliente"):
        g1, g2, g3 = st.columns(3)
        zona_n = g1.selectbox("Zona", C.ZONES)
        cat_n = g2.selectbox("Categoria preferita", C.CATEGORIES)
        piano_n = g3.selectbox("Piano", C.PLANS)
        h1, h2, h3, h4 = st.columns(4)
        iscr = h1.date_input("Data di iscrizione", value=pd.Timestamp(C.REFERENCE_DATE).date() - pd.Timedelta(days=200),
                             max_value=pd.Timestamp(C.REFERENCE_DATE).date(),
                             help=f"Non successiva alla data di riferimento del database ({C.REFERENCE_DATE}).")
        n_req = h2.number_input("Richieste negli ultimi 90 giorni", min_value=0, max_value=60, value=3,
                                help=FIELD_HELP["richieste"])
        amount = h3.number_input("Importo speso negli ultimi 90 giorni (EUR)", min_value=0.0, max_value=5000.0,
                                 value=120.0, step=10.0, help=FIELD_HELP["importo"])
        days = h4.number_input("Giorni dall'ultima richiesta", min_value=0, max_value=365, value=20,
                               help=FIELD_HELP["giorni"])
        submitted = st.form_submit_button("Calcola segmento")

    if submitted:
        rec = dict(zona=zona_n, data_iscrizione=pd.Timestamp(iscr), richieste_90g=int(n_req),
                   importo_totale_90g=float(amount), giorni_ultima_richiesta=float(days),
                   categoria_preferita=cat_n, piano=piano_n)
        errors, warns = N.validate_new_client(rec)
        if errors:
            st.session_state["pending"] = None
            for e in errors:
                st.error(e)
        else:
            exp = seg.explain(pd.DataFrame([rec]))
            st.session_state["pending"] = dict(rec=rec, exp=exp, warns=warns)
        st.session_state["saved"] = None

    pend = st.session_state.get("pending")
    if pend:
        rec, exp = pend["rec"], pend["exp"]
        st.subheader("Riepilogo")
        st.table(pd.DataFrame([{
            "Zona": rec["zona"], "Categoria": rec["categoria_preferita"], "Piano": rec["piano"],
            "Iscrizione": rec["data_iscrizione"].date().isoformat(), "Richieste 90g": rec["richieste_90g"],
            "Importo 90g (EUR)": rec["importo_totale_90g"], "Giorni dall'ultima richiesta": int(rec["giorni_ultima_richiesta"]),
        }]))
        st.success(f"Segmento assegnato: **{exp['segmento']}**")
        st.write(ACTIONS.get(exp["segmento"], ""))
        for w in pend["warns"]:
            st.warning(w)
        if exp["secondo"]:
            if exp["incerto"]:
                st.warning(f"Assegnazione poco netta: il cliente è quasi altrettanto vicino a «{exp['secondo']}».")
            else:
                st.caption(f"Secondo segmento più vicino: {exp['secondo']} (la distanza dal segmento assegnato è "
                           f"{exp['rapporto']:.0%} di quella dal secondo).")
        show_fig(P.scatter_fig(data, "giorni_ultima_richiesta", "importo_totale_90g", cmap,
                                new_point=(rec["giorni_ultima_richiesta"], rec["importo_totale_90g"])))
        s1, s2, _sp = st.columns([1, 1, 3])
        if s1.button("Conferma e salva il cliente", type="primary"):
            row = N.append_new(rec, exp["segmento"])
            st.session_state["pending"] = None
            st.session_state["saved"] = row["customer_id"]
            st.rerun()
        if s2.button("Annulla"):
            st.session_state["pending"] = None
            st.rerun()
    elif st.session_state.get("saved"):
        st.success(f"Cliente {st.session_state['saved']} salvato in data/clienti_nuovi.csv. "
                   "Sarà incluso nel modello solo dopo un riaddestramento (sezione qui sotto).")

    # ---- clienti inseriti e riaddestramento controllato
    st.divider()
    st.subheader("Clienti inseriti e aggiornamento del modello")
    new_all = N.load_new()
    waiting = len(new_all) - n_new
    st.write(f"Clienti inseriti finora: **{len(new_all)}**. Inclusi nel modello in uso: **{n_new}**. "
             f"In attesa di riaddestramento: **{waiting}**.")
    if len(new_all):
        shown = new_all.assign(incluso_nel_modello=[i < n_new for i in range(len(new_all))])
        tabella(shown, height=min(260, 60 + 35 * len(shown)))
        st.download_button("Scarica i nuovi clienti (CSV)", new_all.to_csv(index=False).encode("utf-8"),
                           file_name="clienti_nuovi.csv", mime="text/csv")
    st.caption("Il riaddestramento non è automatico: i segmenti restano stabili finché non lo decidi e prima di "
               "adottare il nuovo modello puoi vedere cosa cambierebbe.")
    r1, r2, _sp2 = st.columns([1, 1, 3])
    if r1.button("Mostra l'effetto del riaddestramento", disabled=waiting <= 0,
                 help="Addestra un modello di prova includendo i nuovi clienti, senza adottarlo." if waiting > 0
                 else "Non ci sono nuovi clienti in attesa."):
        df_all, _ = N.training_data(len(new_all))
        st.session_state["cmp"] = dict(res=N.retrain_comparison(df_all, seg, algo, k), n=len(new_all))
    if r2.button("Torna al modello originale", disabled=n_new == 0,
                 help="Esclude di nuovo tutti i nuovi clienti dall'addestramento."): # non serve cancellare i clienti inseriti, solo il modello torna a quello precedente
        N.set_active_count(0)
        st.session_state["cmp"] = None
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    cmp_ = st.session_state.get("cmp")
    if cmp_ and cmp_["n"] > n_new:
        res = cmp_["res"]
        st.write(f"Modello riaddestrato con {res['clienti_aggiunti']} clienti in più (stessi algoritmo e k):")
        m1, m2, m3 = st.columns(3)
        m1.metric("Silhouette", f"{res['silhouette_dopo']:.3f}", f"{res['silhouette_dopo'] - res['silhouette_prima']:+.3f}",
                  help="Differenza rispetto al modello in uso.")
        m2.metric("Clienti che cambiano segmento", f"{res['quota_cambiata_pct']:.1f}%",
                  help="Quota dei clienti già presenti il cui segmento ha un nome diverso nel nuovo modello.")
        m3.metric("Accordo tra le due partizioni (ARI)", f"{res['ari']:.3f}",
                  help="1 = stessi gruppi; valori bassi = i gruppi sono cambiati molto.")
        tabella(res["tabella"])
        if st.button("Adotta il nuovo modello", type="primary"):
            N.set_active_count(cmp_["n"])
            st.session_state["cmp"] = None
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
