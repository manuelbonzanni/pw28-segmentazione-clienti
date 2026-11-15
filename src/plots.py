"""Grafici matplotlib per grafici di app.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

# palette categorica color-blind safe 
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]
LABELS = {
    "richieste_90g": "Richieste negli ultimi 90 giorni",
    "importo_totale_90g": "Importo speso negli ultimi 90 giorni (EUR)",
    "giorni_ultima_richiesta": "Giorni dall'ultima richiesta",
    "anzianita_giorni": "Anzianità (giorni dall'iscrizione)",
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 110})


def color_map(names):
    """Un colore fisso per segmento che non cambia con i filtri"""
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(sorted(set(names)))}


def elbow_fig(scan, k_sel=None):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(scan["k"], scan["inertia"], "o-", color=PALETTE[0])
    ax[0].set(xlabel="Numero di gruppi k", ylabel="Inerzia (WCSS)", title="Metodo del gomito")
    ax[1].plot(scan["k"], scan["silhouette"], "o-", color=PALETTE[1])
    ax[1].set(xlabel="Numero di gruppi k", ylabel="Indice di silhouette", title="Silhouette media")
    if k_sel is not None:
        for a in ax:
            a.axvline(k_sel, color=PALETTE[4], lw=1.2, alpha=.7)
        ax[0].text(k_sel + .1, ax[0].get_ylim()[1] * .92, f"k = {k_sel}", color="#222222", fontsize=9)
    fig.tight_layout()
    return fig


MARKERS = ["o", "s", "^", "D", "v", "P", "X"]   # codifica secondaria -> forma e non solo colore


def shape_map(names):
    return {n: MARKERS[i % len(MARKERS)] for i, n in enumerate(sorted(set(names)))}


def scatter_fig(df, xcol, ycol, cmap, new_point=None, logy=False, highlight=None):
    """Scatter per segmento, con marcatori diversi e possibilità di evidenziarne uno (highlight)"""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    shapes = shape_map(cmap.keys())
    handles = []
    names = sorted(df["segmento"].unique())
    # prima gli sfondi grigi, poi il segmento evidenziato (cosi' resta sopra)
    order = [n for n in names if highlight and n != highlight] + [n for n in names if not highlight or n == highlight]
    for name in order:
        d = df[df["segmento"] == name]
        if highlight and name != highlight:
            ax.scatter(d[xcol], d[ycol], s=16, alpha=.45, color="#C8C8C8", marker="o", linewidths=0, zorder=1)
            continue
        ax.scatter(d[xcol], d[ycol], s=26, alpha=.75, color=cmap[name], marker=shapes[name],
                   edgecolor="white", linewidths=.5, zorder=2)
        handles.append(Line2D([], [], marker=shapes[name], ls="", color=cmap[name], markeredgecolor="white",
                              markersize=8, label=f"{name} ({len(d)})"))
    if highlight:
        handles.append(Line2D([], [], marker="o", ls="", color="#C8C8C8", markersize=6, label="Altri segmenti"))
    if new_point is not None:
        ax.scatter([new_point[0]], [new_point[1]], s=230, marker="*", color="black", edgecolor="white",
                   linewidths=.9, zorder=5)
        handles.append(Line2D([], [], marker="*", ls="", color="black", markersize=12, label="Nuovo cliente"))
    ax.set(xlabel=LABELS.get(xcol, xcol), ylabel=LABELS.get(ycol, ycol))
    if logy:
        ax.set_yscale("symlog")
    ax.margins(x=.03, y=.05)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2 if len(handles) > 3 else len(handles),
              frameon=False, fontsize=9, columnspacing=1.6, handletextpad=.4)
    fig.tight_layout()
    return fig


SHORT = {"Clienti fedeli ad alto valore": "Fedeli", "Clienti regolari": "Regolari", "Clienti occasionali": "Occasionali",
         "Nuovi clienti in fase di prova": "Nuovi / prova", "Inattivi / a rischio abbandono": "Inattivi"}


def short_name(name):
    return SHORT.get(name, name)


def describe_scatter(df, xcol, ycol):
    """Testo alternativo del grafico mediane per segmento"""
    parts = []
    for name, d in df.groupby("segmento"):
        parts.append(f"{short_name(name)}: {len(d)} clienti, mediana {LABELS.get(xcol, xcol).split(' (')[0].lower()} "
                     f"{d[xcol].median():.0f}, mediana {LABELS.get(ycol, ycol).split(' (')[0].lower()} {d[ycol].median():.0f}")
    return "; ".join(parts) + "."


def profile_heatmap(profile, cols):
    """Valori medi per segmento, colorati come scostamento dalla media"""
    M = profile[cols].astype(float)
    Z = (M - M.mean()) / M.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(7.2, 0.7 * len(M) + 1.4))
    im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(cols)), [LABELS.get(c, c).split(" (")[0].replace(" negli ultimi 90 giorni", " 90g").replace(" speso negli ultimi 90 giorni", " 90g")
                                      for c in cols], rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(M)), profile["segmento"], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.iloc[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(Z.iloc[i, j]) > 1.2 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=.8, label="scostamento dalla media (z)")
    fig.tight_layout()
    return fig


def size_revenue_fig(profile, cmap):
    """Quota di clienti e quota di ricavi per ogni segmento"""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    p = profile.sort_values("quota_ricavi_%")
    y = np.arange(len(p))
    fmt = lambda v: f"{v:.1f}%".replace(".", ",")
    ax.barh(y + .2, p["quota_clienti_%"], .38, color="#B0B0B0", edgecolor="white", lw=1.5)
    ax.barh(y - .2, p["quota_ricavi_%"], .38, color=[cmap[s] for s in p["segmento"]], edgecolor="white", lw=1.5)
    for yi, v in zip(y + .2, p["quota_clienti_%"]):
        ax.text(v + .8, yi, fmt(v), va="center", fontsize=8.5, color="#666666")
    for yi, v in zip(y - .2, p["quota_ricavi_%"]):
        ax.text(v + .8, yi, fmt(v), va="center", fontsize=8.5, color="#222222", weight="bold")
    ax.set_yticks(y, [short_name(n) for n in p["segmento"]], fontsize=9)
    ax.set_xlim(0, max(p["quota_clienti_%"].max(), p["quota_ricavi_%"].max()) * 1.18)
    ax.set(xlabel="% sul totale")
    fig.tight_layout()
    return fig


def silhouette_plot(X, labels, names, cmap):
    s = silhouette_samples(X, labels)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y0 = 0
    for c in sorted(set(labels)):
        v = np.sort(s[labels == c])
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=cmap[names[c]], alpha=.85)
        ax.text(-0.05, y0 + len(v) / 2, short_name(names[c]), ha="right", va="center", fontsize=9)
        y0 += len(v) + 10
    ax.axvline(s.mean(), color="black", ls="--", lw=1)
    ax.set(xlabel="Coefficiente di silhouette per cliente", yticks=[])
    ax.set_xlim(-0.4, 1)
    fig.tight_layout()
    return fig


# palette adatta al daltonismo x le cinque categorie di servizio
CATEGORY_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]

def zone_category_fig(df):
    """Barre orizzontali al 100% -> per ogni zona, quota dei clienti per categoria preferita"""
    ct = pd.crosstab(df["zona"], df["categoria_preferita"])
    ct = ct.reindex(index=C.ZONES, columns=C.CATEGORIES, fill_value=0)
    totals = ct.sum(axis=1).replace(0, np.nan)
    share = ct.div(totals, axis=0).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    left = np.zeros(len(share))
    y = np.arange(len(share))
    for cat, col in zip(C.CATEGORIES, CATEGORY_COLORS):
        vals = share[cat].to_numpy()
        ax.barh(y, vals, left=left, color=col, edgecolor="white", label=cat, height=0.7)
        for yi, v, l in zip(y, vals, left):
            if v >= 6:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if col not in ("#E69F00", "#56B4E9") else "black")
        left = left + vals
    ax.set_yticks(y)
    ax.set_yticklabels([f"{z} (n={int(totals[z]) if not np.isnan(totals[z]) else 0})" for z in share.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% dei clienti della zona")
    ax.set_title("Categoria preferita per zona", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.45, -0.18), frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
