"""
Optimisateur de Puissance Souscrite TURPE 7
Interface Streamlit — Enedis uniquement
"""

import io
import base64
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from turpe_engine import (
    charger_fichier_auto,
    resumer_chargement,
    classifier_dataframe,
    calculer_cout_total,
    optimiser_puissances,
    PLAGES_HTA, PLAGES_BT_SUP,
    HTA_BI, BT_SUP_BI,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Optimisateur de Facture Électrique", page_icon="⚡", layout="wide")

st.title("⚡ Optimisateur de Facture Électrique")
st.caption("TURPE 7 + CTA — Réseau Enedis | Tarifs en vigueur au 1er février 2026 | Montants HT")
st.divider()

COULEURS_PLAGES = {
    "Pointe": "#FF4444", "HPH": "#FF8C00", "HCH": "#FFD700",
    "HPB": "#4CAF50",    "HCB": "#2196F3", "unique": "#9C27B0",
}


# ─────────────────────────────────────────────
# HELPER : figure → image bytes (pour PDF)
# ─────────────────────────────────────────────
def _mpl_courbe_charge(df, couleurs) -> bytes:
    """Courbe de charge colorée par plage — matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.2))
    for plage in sorted(df["plage"].unique()):
        df_p = df[df["plage"] == plage]
        ax.scatter(df_p["timestamp"], df_p["puissance_kw"],
                   s=0.8, color=couleurs.get(plage, "#888"), label=plage, alpha=0.8)
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Puissance (kW)", fontsize=8)
    ax.set_title("Courbe de charge par plage horosaisonnière", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", markerscale=4)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def _mpl_composantes(composantes, actuel, optimal, intermediaire=None, label_inter="PS opt.") -> bytes:
    """Graphique barres composantes TURPE — matplotlib. Supporte 2 ou 3 séries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(composantes))
    if intermediaire is not None:
        w = 0.25
        fig, ax = plt.subplots(figsize=(9, 3.0))
        ax.bar(x - w,   actuel,        w, label="Actuel",       color="#FF6B6B")
        ax.bar(x,       intermediaire, w, label=label_inter,     color="#FFB74D")
        ax.bar(x + w,   optimal,       w, label="Optimal",       color="#4CAF50")
    else:
        w = 0.35
        fig, ax = plt.subplots(figsize=(9, 3.0))
        ax.bar(x - w/2, actuel,  w, label="Actuel",   color="#FF6B6B")
        ax.bar(x + w/2, optimal, w, label="Optimisé", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(composantes, fontsize=9)
    ax.set_ylabel("€/an", fontsize=8)
    ax.set_title("Composantes TURPE : actuel vs optimisé", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def _mpl_projection(nb_annees, eco_annuelle) -> bytes:
    """Graphique projection pluriannuelle — matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    annees    = list(range(1, nb_annees + 1))
    eco_cumul = [eco_annuelle * a for a in annees]

    fig, ax1 = plt.subplots(figsize=(9, 3.0))
    ax2 = ax1.twinx()
    ax1.bar(annees, [eco_annuelle] * nb_annees, color="#4CAF50", alpha=0.7, label="Économie annuelle")
    ax2.plot(annees, eco_cumul, color="#1565C0", marker="o", markersize=3, linewidth=2, label="Cumul")
    ax1.set_xlabel("Année", fontsize=8)
    ax1.set_ylabel("Économie annuelle (€)", fontsize=8, color="#4CAF50")
    ax2.set_ylabel("Économie cumulée (€)", fontsize=8, color="#1565C0")
    ax1.set_title(f"Projection des économies sur {nb_annees} ans", fontsize=9)
    ax1.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


# ─────────────────────────────────────────────
# HELPER : génération PDF avec reportlab
# ─────────────────────────────────────────────
def generer_pdf(
    df_raw, df, nom_etude, domaine, fta, fta_opt, type_contrat,
    hc_debut, hc_fin,
    ps_actuelles, resultat_actuel, resultat_optimal,
    economie, economie_pct, economie_cta, economie_cta_pct,
    nb_annees, resultats_fta,
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable, PageBreak,
    )
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
    )

    W, H = A4
    content_w = W - 4*cm

    # ── Styles ────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    BLEU  = colors.HexColor("#1565C0")
    GRIS  = colors.HexColor("#F5F5F5")
    VERT  = colors.HexColor("#2E7D32")
    ROUGE = colors.HexColor("#C62828")

    s_titre      = ParagraphStyle("titre",    fontSize=16, textColor=BLEU,  spaceAfter=8,  fontName="Helvetica-Bold")
    s_sous_titre = ParagraphStyle("soustitre",fontSize=8,  textColor=colors.HexColor("#455A64"), spaceAfter=10, fontName="Helvetica")
    s_h2         = ParagraphStyle("h2",       fontSize=11, textColor=BLEU,  spaceBefore=12, spaceAfter=5, fontName="Helvetica-Bold")
    s_date       = ParagraphStyle("date",     fontSize=8,  textColor=colors.HexColor("#78909C"), alignment=TA_RIGHT, fontName="Helvetica")
    s_kpi_label  = ParagraphStyle("kpilbl",   fontSize=8,  textColor=colors.HexColor("#546E7A"), alignment=TA_CENTER, fontName="Helvetica")
    s_kpi_val    = ParagraphStyle("kpival",   fontSize=15, textColor=BLEU,  alignment=TA_CENTER, fontName="Helvetica-Bold")
    s_kpi_eco    = ParagraphStyle("kpieco",   fontSize=15, textColor=VERT,  alignment=TA_CENTER, fontName="Helvetica-Bold")
    s_kpi_neg    = ParagraphStyle("kpineg",   fontSize=15, textColor=ROUGE, alignment=TA_CENTER, fontName="Helvetica-Bold")
    s_cell       = ParagraphStyle("cell",     fontSize=8,  fontName="Helvetica", wordWrap="CJK")
    s_cell_bold  = ParagraphStyle("cellbold", fontSize=8,  fontName="Helvetica-Bold", wordWrap="CJK")
    s_footer     = ParagraphStyle("footer",   fontSize=7,  textColor=colors.HexColor("#90A4AE"), alignment=TA_CENTER, fontName="Helvetica")

    # Variables dérivées
    fta_change           = fta_opt != fta
    resultat_fta_act_pdf = resultats_fta.get(fta, {}).get("resultat", resultat_actuel)

    story = []

    # ── EN-TÊTE ───────────────────────────────────────────────────────────────
    story.append(Paragraph(nom_etude, s_titre))
    fta_label = f"FTA recommandée : {fta_opt}" if fta_opt != fta else f"FTA : {fta}"
    story.append(Paragraph(
        f"Optimisation TURPE 7 + CTA — {domaine} | {fta_label} | {type_contrat.replace('_', ' ').title()} — Montants HT",
        s_sous_titre
    ))
    story.append(Paragraph(f"Rapport du {datetime.now().strftime('%d/%m/%Y')}", s_date))
    story.append(HRFlowable(width="100%", thickness=2, color=BLEU, spaceAfter=10))

    # ── INFOS SITE ────────────────────────────────────────────────────────────
    story.append(Paragraph("Informations du site", s_h2))
    nb_jours = df_raw.attrs.get("nb_jours", 365)
    debut_str = df_raw.attrs.get("periode_debut", "?")
    fin_str   = df_raw.attrs.get("periode_fin",   "?")
    debut_str = debut_str.strftime("%d/%m/%Y") if hasattr(debut_str, "strftime") else str(debut_str)[:10]
    fin_str   = fin_str.strftime("%d/%m/%Y")   if hasattr(fin_str,   "strftime") else str(fin_str)[:10]

    data_site = [
        ["PRM",            str(df_raw.attrs.get("prm", "—")),            "Période analysée", f"{debut_str} → {fin_str}"],
        ["Durée fichier",  f"{nb_jours} j ({round(nb_jours/365*100,1)} %)", "Plages HC",     f"{hc_debut}h → {hc_fin}h (7j/7)"],
        ["Puissance max",  f"{round(df['puissance_kw'].max(),1)} kW",    "Puissance moy.",   f"{round(df['puissance_kw'].mean(),1)} kW"],
    ]
    t_site = Table(data_site, colWidths=[3*cm, 5.5*cm, 3*cm, 5.5*cm])
    t_site.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME",  (0,0), (0,-1),  "Helvetica-Bold"),
        ("FONTNAME",  (2,0), (2,-1),  "Helvetica-Bold"),
        ("FONTSIZE",  (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, GRIS]),
        ("GRID",      (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t_site)
    story.append(Spacer(1, 8))

    # ── KPIs ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Résultats de l'optimisation (TURPE + CTA — HT)", s_h2))
    s_eco = s_kpi_eco if economie_cta >= 0 else s_kpi_neg

    if fta_change:
        eco_ps  = resultat_actuel["Total_HT"] - resultat_fta_act_pdf["Total_HT"]
        eco_fta = resultat_fta_act_pdf["Total_HT"] - resultat_optimal["Total_HT"]
        kpi_data = [
            [Paragraph("Situation actuelle",         s_kpi_label),
             Paragraph(f"PS opt. FTA act. ({fta})", s_kpi_label),
             Paragraph(f"Optimal ({fta_opt})",       s_kpi_label),
             Paragraph("Économie totale",             s_kpi_label)],
            [Paragraph(f"{resultat_actuel['Total_HT']:,.0f} €/an",         s_kpi_val),
             Paragraph(f"{resultat_fta_act_pdf['Total_HT']:,.0f} €/an",    s_kpi_val),
             Paragraph(f"{resultat_optimal['Total_HT']:,.0f} €/an",        s_kpi_val),
             Paragraph(f"-{abs(economie_cta):,.0f} €/an\n({economie_cta_pct:.1f} %)", s_eco)],
        ]
        col_w = content_w / 4
    else:
        kpi_data = [
            [Paragraph("Coût actuel HT",  s_kpi_label), Paragraph("Coût optimisé HT", s_kpi_label),
             Paragraph("Économie HT/an",  s_kpi_label), Paragraph("Gain relatif",     s_kpi_label)],
            [Paragraph(f"{resultat_actuel['Total_HT']:,.0f} €/an",  s_kpi_val),
             Paragraph(f"{resultat_optimal['Total_HT']:,.0f} €/an", s_kpi_val),
             Paragraph(f"-{abs(economie_cta):,.0f} €/an", s_eco),
             Paragraph(f"{economie_cta_pct:.1f} %",       s_eco)],
        ]
        col_w = content_w / 4

    t_kpi = Table(kpi_data, colWidths=[col_w]*4)
    t_kpi.setStyle(TableStyle([
        ("BOX",        (0,0), (-1,-1), 1, BLEU),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, colors.HexColor("#CFD8DC")),
        ("BACKGROUND", (0,0), (-1,0),  colors.HexColor("#E3F2FD")),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 8))

    # ── TABLEAU PS ───────────────────────────────────────────────────────────
    story.append(Paragraph("Puissances souscrites recommandées", s_h2))
    ps_opt_act = resultat_fta_act_pdf["puissances_souscrites"]
    ps_opt_opt = resultat_optimal["puissances_souscrites"]

    if fta_change:
        rows_ps = [[Paragraph(h, s_cell_bold) for h in
                    ["Plage", "Actuelle", f"PS opt. ({fta})", f"Optimal ({fta_opt})"]]]
        for p in ps_actuelles:
            ecart_final = ps_opt_opt.get(p, 0) - ps_actuelles[p]
            coul = VERT if ecart_final < 0 else (ROUGE if ecart_final > 0 else colors.black)
            rows_ps.append([
                Paragraph(p, s_cell_bold),
                Paragraph(f"{ps_actuelles[p]} kVA",      s_cell),
                Paragraph(f"{ps_opt_act.get(p,0)} kVA",  s_cell),
                Paragraph(f"{ps_opt_opt.get(p,0)} kVA",
                          ParagraphStyle("e", fontSize=8, textColor=coul, fontName="Helvetica-Bold")),
            ])
        t_ps = Table(rows_ps, colWidths=[2.5*cm, 3.2*cm, 3.5*cm, 3.5*cm])
    else:
        rows_ps = [[Paragraph(h, s_cell_bold) for h in ["Plage", "Actuelle", "Optimisée", "Écart"]]]
        for p in ps_actuelles:
            ecart = ps_opt_opt.get(p, 0) - ps_actuelles[p]
            coul  = VERT if ecart < 0 else (ROUGE if ecart > 0 else colors.black)
            rows_ps.append([
                Paragraph(p, s_cell_bold),
                Paragraph(f"{ps_actuelles[p]} kVA", s_cell),
                Paragraph(f"{ps_opt_opt.get(p, '—')} kVA", s_cell),
                Paragraph(f"{'+' if ecart > 0 else ''}{ecart} kVA",
                          ParagraphStyle("e", fontSize=8, textColor=coul, fontName="Helvetica-Bold")),
            ])
        t_ps = Table(rows_ps, colWidths=[3*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    t_ps.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLEU),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GRIS]),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
        ("ALIGN",      (1,0), (-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t_ps)
    story.append(Spacer(1, 8))

    # ── TABLEAU COMPARATIF FTA ────────────────────────────────────────────────
    if len(resultats_fta) > 1:
        story.append(Paragraph("Comparaison des formules tarifaires (PS optimisées)", s_h2))
        rows_fta_pdf = [[Paragraph(h, s_cell_bold) for h in
                         ["FTA", "PS optimisées", "TURPE HT", "CTA HT", "Total HT", "Économie vs actuel"]]]
        for fta_k, v in sorted(resultats_fta.items(), key=lambda x: x[1]["resultat"]["Total_HT"]):
            r      = v["resultat"]
            ps     = r["puissances_souscrites"]
            ps_str = " / ".join(f"{p}:{int(ps[p])} kVA" for p in ps) if len(ps) > 1 else f"{int(list(ps.values())[0])} kVA"
            ecart  = round(resultat_actuel["Total_HT"] - r["Total_HT"], 0)
            is_best = fta_k == fta_opt
            is_cur  = fta_k == fta
            # Remplacement emojis par texte lisible avec Helvetica
            suf     = " [OPT.]" if is_best else (" [act.]" if is_cur else "")
            style_l = s_cell_bold if is_best else s_cell
            rows_fta_pdf.append([
                Paragraph(fta_k + suf, style_l),
                Paragraph(ps_str, s_cell),
                Paragraph(f"{r['Total']:,.0f} €",    style_l),
                Paragraph(f"{r['CTA_HT']:,.0f} €",   style_l),
                Paragraph(f"{r['Total_HT']:,.0f} €", style_l),
                Paragraph(f"{'+' if ecart>=0 else ''}{ecart:,.0f} €", style_l),
            ])
        t_fta = Table(rows_fta_pdf, colWidths=[3.2*cm, 3.8*cm, 2.4*cm, 2.2*cm, 2.4*cm, 2.7*cm])
        t_fta.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0),  BLEU),
            ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GRIS]),
            ("BACKGROUND",     (0,1), (-1,1),  colors.HexColor("#C8E6C9")),  # meilleure (triée 1re)
            ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
            ("ALIGN",          (2,0), (-1,-1), "RIGHT"),
            ("TOPPADDING",     (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 3),
        ]))
        story.append(t_fta)
        story.append(Spacer(1, 8))

    # ── TABLEAU COMPOSANTES ───────────────────────────────────────────────────
    story.append(Paragraph("Détail des composantes TURPE + CTA — HT annualisés", s_h2))
    compo_list   = ["CG", "CC", "CS", "CMDPS", "CTA_HT", "Total_HT"]
    labels_compo = {"CG": "Gestion (CG)", "CC": "Comptage (CC)", "CS": "Soutirage (CS)",
                    "CMDPS": "Dépassement (CMDPS)", "CTA_HT": "CTA HT (15 %)", "Total_HT": "TOTAL HT"}

    if fta_change:
        rows_c = [[Paragraph(h, s_cell_bold) for h in
                   ["Composante", "Actuel", f"PS opt. ({fta})", f"Optimal ({fta_opt})", "Écart total"]]]
        for c in compo_list:
            act     = resultat_actuel.get(c, 0)
            inter   = resultat_fta_act_pdf.get(c, 0)
            opt     = resultat_optimal.get(c, 0)
            ecart_c = opt - act
            is_tot  = c == "Total_HT"
            sl      = s_cell_bold if is_tot else s_cell
            rows_c.append([
                Paragraph(labels_compo[c], sl),
                Paragraph(f"{act:,.0f} €/an",                                           sl),
                Paragraph(f"{inter:,.0f} €/an",                                         sl),
                Paragraph(f"{opt:,.0f} €/an",                                           sl),
                Paragraph(f"{'+' if ecart_c > 0 else ''}{ecart_c:,.0f} €/an",           sl),
            ])
        t_comp = Table(rows_c, colWidths=[3.5*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.8*cm])
    else:
        rows_c = [[Paragraph(h, s_cell_bold) for h in ["Composante", "Actuel", "Optimisé", "Écart"]]]
        for c in compo_list:
            act     = resultat_actuel.get(c, 0)
            opt     = resultat_optimal.get(c, 0)
            ecart_c = opt - act
            is_tot  = c == "Total_HT"
            sl      = s_cell_bold if is_tot else s_cell
            rows_c.append([
                Paragraph(labels_compo[c], sl),
                Paragraph(f"{act:,.0f} €/an",                                 sl),
                Paragraph(f"{opt:,.0f} €/an",                                 sl),
                Paragraph(f"{'+' if ecart_c > 0 else ''}{ecart_c:,.0f} €/an", sl),
            ])
        t_comp = Table(rows_c, colWidths=[4*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    style_comp = [
        ("BACKGROUND",  (0,0), (-1,0),  BLEU),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, GRIS]),
        ("BACKGROUND",  (0, len(compo_list)), (-1, len(compo_list)), colors.HexColor("#E3F2FD")),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
        ("ALIGN",       (1,0), (-1,-1), "RIGHT"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ]
    t_comp.setStyle(TableStyle(style_comp))
    story.append(t_comp)

    # ── GRAPHIQUES ────────────────────────────────────────────────────────────
    from reportlab.platypus import KeepTogether

    story.append(PageBreak())
    story.append(Paragraph(f"{nom_etude} — {datetime.now().strftime('%d/%m/%Y')}", s_date))
    story.append(HRFlowable(width="100%", thickness=1, color=BLEU, spaceAfter=8))

    png_courbe = _mpl_courbe_charge(df, COULEURS_PLAGES)
    story.append(KeepTogether([
        Paragraph("Courbe de charge par plage horosaisonnière", s_h2),
        RLImage(io.BytesIO(png_courbe), width=content_w, height=content_w*3.2/9),
    ]))
    story.append(Spacer(1, 8))

    compo_graph_pdf  = ["CG", "CC", "CS", "CMDPS", "CTA_HT"]
    labels_graph_pdf = ["Gestion", "Comptage", "Soutirage", "Dépassement", "CTA HT"]
    png_compo = _mpl_composantes(
        labels_graph_pdf,
        [resultat_actuel[c]        for c in compo_graph_pdf],
        [resultat_optimal[c]       for c in compo_graph_pdf],
        intermediaire=[resultat_fta_act_pdf[c] for c in compo_graph_pdf] if fta_change else None,
        label_inter=f"PS opt. ({fta})",
    )
    story.append(KeepTogether([
        Paragraph("TURPE + CTA HT annualisé : actuel vs optimisé par composante", s_h2),
        RLImage(io.BytesIO(png_compo), width=content_w, height=content_w*3.0/9),
    ]))
    story.append(Spacer(1, 8))

    png_proj = _mpl_projection(nb_annees, max(0, economie_cta))
    story.append(KeepTogether([
        Paragraph(f"Économie annuelle HT — cumul sur {nb_annees} ans : {max(0, economie_cta) * nb_annees:,.0f} €", s_h2),
        RLImage(io.BytesIO(png_proj), width=content_w, height=content_w*3.0/9),
    ]))

    # ── PIED DE PAGE ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#B0BEC5")))
    story.append(Paragraph(
        "TURPE 7 + CTA — Réseau Enedis — Tarifs au 1er février 2026 — Délibération CRE n°2025-78 — Montants hors TVA (HT)",
        s_footer
    ))

    doc.build(story)
    return buffer.getvalue()


# ─────────────────────────────────────────────
# SIDEBAR — PARAMÉTRAGE
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Paramétrage")

    nom_etude = st.text_input("Nom de l'étude", value="Étude d'optimisation",
                              help="Apparaîtra en titre sur le rapport PDF")

    domaine = st.selectbox("Domaine de tension", ["HTA", "BT > 36 kVA", "BT ≤ 36 kVA"])

    if domaine == "HTA":
        fta_options = ["CU pointe fixe", "CU pointe mobile", "LU pointe fixe", "LU pointe mobile"]
    elif domaine == "BT > 36 kVA":
        fta_options = ["CU", "LU"]
    else:
        fta_options = ["CU4", "MU4", "LU", "CU (dérogatoire)", "MUDT (dérogatoire)"]
    fta = st.selectbox("Formule Tarifaire (FTA)", fta_options)

    type_contrat = st.selectbox(
        "Type de contrat", ["contrat_unique", "CARD"],
        format_func=lambda x: "Contrat unique" if x == "contrat_unique" else "CARD (direct Enedis)",
    )

    st.divider()

    # ── Heures Creuses ────────────────────────
    st.subheader("🌙 Heures Creuses")
    st.caption("HC s'appliquent 7j/7")

    hc_debut = st.slider("Début HC", min_value=0, max_value=23, value=22,
                         help="Heure de début des Heures Creuses (incluse)")
    hc_fin   = st.slider("Fin HC",   min_value=0, max_value=23, value=6,
                         help="Heure de fin des Heures Creuses (exclue)")

    if hc_debut > hc_fin:
        nb_hc = (24 - hc_debut) + hc_fin
    else:
        nb_hc = hc_fin - hc_debut
    nb_hp = 24 - nb_hc
    st.info(f"HC : **{nb_hc}h**/jour ({hc_debut}h → {hc_fin}h)\nHP : **{nb_hp}h**/jour")

    st.divider()

    # ── Puissances actuelles ──────────────────
    st.subheader("📋 Puissances actuelles (kVA)")
    st.caption("Contrainte TURPE : HPH ≤ HCH ≤ HPB ≤ HCB")

    plages = PLAGES_HTA if domaine == "HTA" else PLAGES_BT_SUP if domaine == "BT > 36 kVA" else ["unique"]

    ps_identiques = st.checkbox("Même puissance pour toutes les plages", value=False)

    ps_actuelles = {}
    if ps_identiques:
        ps_commune = st.number_input("PS unique (kVA)", min_value=1, max_value=10000, value=100, step=1)
        for plage in plages:
            ps_actuelles[plage] = ps_commune
        st.caption(f"→ {', '.join(plages)} = {ps_commune} kVA")
    else:
        for plage in plages:
            ps_actuelles[plage] = st.number_input(
                f"PS {plage} (kVA)", min_value=1, max_value=10000, value=100, step=1, key=f"ps_{plage}"
            )

    pas_kva = 1  # Pas de balayage fixé à 1 kVA

    st.divider()
    # ── Optimisation FTA ─────────────────────
    st.subheader("⚙️ Options d'optimisation")
    optimiser_fta = st.checkbox(
        "Comparer toutes les FTA",
        value=False,
        help="Teste toutes les formules tarifaires du domaine et retient "
             "celle qui minimise TURPE + CTA HT. Les PS sont optimisées "
             "pour chaque FTA candidate. Prix de fourniture non affectés.",
    )


# ─────────────────────────────────────────────
# IMPORT DES DONNÉES
# ─────────────────────────────────────────────
st.header("📂 Import de la courbe de charge")

col_import, col_format = st.columns([2, 1])
with col_import:
    uploaded_file = st.file_uploader(
        "Importez votre courbe de charge (CSV)",
        type=["csv"],
        help="Format Enedis R63 ou Likewatt — détection automatique"
    )
with col_format:
    st.info("""
    **Formats acceptés**

    🔵 **Enedis SGE R63** — `Horodate` | `Valeur` (W)

    🟢 **Likewatt** — `Date de la mesure` | `Heure de la mesure` | `PA` (W)

    Détection **automatique** du format.
    Séparateur : **;**  |  Unité : **W**  |  Pas : **5 min**
    """)

if uploaded_file:
    try:
        df_raw, format_detecte = charger_fichier_auto(uploaded_file)
        st.session_state["df_raw"] = df_raw
        st.session_state["format_detecte"] = format_detecte
        resume = resumer_chargement(df_raw)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("📁 Format",    format_detecte)
        c2.metric("📅 Durée",     f"{resume['Nombre de jours']} jours")
        c3.metric("📊 Couverture", resume["Couverture annuelle"])
        c4.metric("🔢 Points",    f"{resume['Points horaires']:,}")
        c5.metric("⚡ Pmax",      f"{resume['Puissance max (kW)']} kW")
        c6.metric("🔄 Facteur",   resume["Facteur d'annualisation"])
        if df_raw.attrs.get("nb_jours", 365) < 90:
            st.warning("⚠️ Moins de 3 mois — résultats extrapolés, à interpréter avec prudence.")
        elif df_raw.attrs.get("nb_jours", 365) < 365:
            st.info(f"ℹ️ {resume['Nombre de jours']} jours disponibles — coûts extrapolés sur 12 mois.")
        else:
            st.success(f"✅ {resume['Nombre de jours']} jours chargés — couverture optimale.")
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        st.stop()
else:
    if st.button("🎲 Générer des données de démonstration (90 jours)", type="secondary"):
        np.random.seed(42)
        dates   = pd.date_range("2025-01-01", periods=90*24*12, freq="5min")
        base    = 20 + 10*np.sin(np.linspace(0, 8*np.pi, len(dates)))
        bruit   = np.random.normal(0, 3, len(dates))
        pics    = np.zeros(len(dates))
        pics[np.random.choice(len(dates), 30, replace=False)] = np.random.uniform(15, 35, 30)
        valeurs = np.clip(base + bruit + pics, 0, 55) * 1000
        csv_demo = pd.DataFrame({
            "Identifiant PRM": "30001234567890", "Date de début": "2025-01-01",
            "Date de fin": "2025-04-01", "Grandeur physique": "PA",
            "Grandeur métier": "CONS", "Etape métier": "BEST", "Unité": "W",
            "Horodate": dates, "Valeur": valeurs.astype(int),
            "Nature": "R", "Pas": "PT5M",
            "Indice de vraisemblance": "null", "Etat complémentaire": "null",
        }).to_csv(sep=";", index=False)
        df_raw, fmt = charger_fichier_auto(io.StringIO(csv_demo))
        st.session_state["df_raw"] = df_raw
        st.session_state["format_detecte"] = fmt
        st.success("✅ Données de démonstration générées")
        st.rerun()

    if "df_raw" not in st.session_state:
        st.info("👆 Importez un fichier CSV ou générez des données de démonstration.")
        st.stop()


# ─────────────────────────────────────────────
# ANALYSE
# ─────────────────────────────────────────────
df_raw = st.session_state["df_raw"]
df     = classifier_dataframe(df_raw, domaine, fta, hc_debut, hc_fin)

st.divider()
st.header("📊 Analyse de la courbe de charge")

fig_courbe = go.Figure()
for plage in sorted(df["plage"].unique()):
    df_p = df[df["plage"] == plage]
    fig_courbe.add_trace(go.Scatter(
        x=df_p["timestamp"], y=df_p["puissance_kw"],
        mode="markers", marker=dict(size=2, color=COULEURS_PLAGES.get(plage, "#888")),
        name=plage,
    ))
fig_courbe.update_layout(
    title=f"Courbe de charge — HC : {hc_debut}h → {hc_fin}h",
    xaxis_title="Date", yaxis_title="Puissance (kW)",
    height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_courbe, use_container_width=True)

stats = df.groupby("plage")["puissance_kw"].agg(
    Heures="count",
    Moy=lambda x: round(x.mean(), 1),
    P90=lambda x: round(x.quantile(0.90), 1),
    P95=lambda x: round(x.quantile(0.95), 1),
    Max=lambda x: round(x.max(), 1),
).rename(columns={"Moy": "Moy (kW)", "P90": "P90 (kW)", "P95": "P95 (kW)", "Max": "Max (kW)"})
stats.index.name = "Plage"
st.dataframe(stats, use_container_width=True)


st.divider()

# ─────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────
st.header("💡 Optimisation — Puissances & Formule Tarifaire")

# ── Calcul du coût actuel ──────────────────────────────────────────────────────
resultat_actuel = calculer_cout_total(df.copy(), domaine, fta, ps_actuelles, type_contrat)

# ── Optimisation ──────────────────────────────────────────────────────────────
FTA_PAR_DOMAINE = {
    "HTA":         ["CU pointe fixe", "CU pointe mobile", "LU pointe fixe", "LU pointe mobile"],
    "BT > 36 kVA": ["CU", "LU"],
    "BT ≤ 36 kVA": ["CU4", "MU4", "LU", "CU (dérogatoire)", "MUDT (dérogatoire)"],
}
# Si cochée : on teste toutes les FTA du domaine ; sinon, uniquement la FTA actuelle
fta_candidates = FTA_PAR_DOMAINE[domaine] if optimiser_fta else [fta]

resultats_fta = {}
spinner_msg = "⏳ Optimisation sur toutes les formules tarifaires..." if optimiser_fta else "⏳ Optimisation en cours..."
with st.spinner(spinner_msg):
    for fta_cand in fta_candidates:
        # Pour HTA, la classification des heures Pointe dépend de la FTA (fixe vs mobile)
        # → on reclassifie pour chaque candidat
        df_cand = classifier_dataframe(df_raw, domaine, fta_cand, hc_debut, hc_fin)
        res_opt_cand, df_sc_cand = optimiser_puissances(
            df_cand, domaine, fta_cand, type_contrat, pas_kva,
            ps_actuelles=ps_actuelles if fta_cand == fta else None,
        )
        resultats_fta[fta_cand] = {
            "resultat":  res_opt_cand,
            "scenarios": df_sc_cand,
            "df_classe": df_cand,
        }

# Meilleure FTA = celle dont le Total_HT optimisé est le plus bas
fta_opt          = min(resultats_fta, key=lambda k: resultats_fta[k]["resultat"]["Total_HT"])
resultat_optimal = resultats_fta[fta_opt]["resultat"]
df_scenarios     = resultats_fta[fta_opt]["scenarios"]
df_opt           = resultats_fta[fta_opt]["df_classe"]   # df classifié avec la FTA optimale
fta_change           = fta_opt != fta
resultat_fta_actuelle = resultats_fta[fta]["resultat"]   # PS optimisées avec la FTA actuelle

economie         = resultat_actuel["Total"]    - resultat_optimal["Total"]
economie_cta     = resultat_actuel["Total_HT"] - resultat_optimal["Total_HT"]
economie_pct     = (economie     / resultat_actuel["Total"]    * 100) if resultat_actuel["Total"]    > 0 else 0
economie_cta_pct = (economie_cta / resultat_actuel["Total_HT"] * 100) if resultat_actuel["Total_HT"] > 0 else 0

# ── KPIs ──────────────────────────────────────────────────────────────────────
if fta_change:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📌 Situation actuelle",
              f"{resultat_actuel['Total_HT']:,.0f} €/an")
    c2.metric(f"🔧 PS opt. FTA act. ({fta})",
              f"{resultat_fta_actuelle['Total_HT']:,.0f} €/an",
              delta=f"{resultat_fta_actuelle['Total_HT'] - resultat_actuel['Total_HT']:+,.0f} €")
    c3.metric(f"✅ Optimal ({fta_opt})",
              f"{resultat_optimal['Total_HT']:,.0f} €/an",
              delta=f"{economie_cta:+,.0f} €")
    c4.metric("📉 Économie totale",
              f"{economie_cta:,.0f} €/an",
              delta=f"{economie_cta_pct:.1f} %")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("📌 Coût TURPE + CTA actuel (HT)",      f"{resultat_actuel['Total_HT']:,.0f} €/an")
    c2.metric("✅ Coût TURPE + CTA optimisé (HT)",    f"{resultat_optimal['Total_HT']:,.0f} €/an",
              delta=f"-{economie_cta:,.0f} €")
    c3.metric("📉 Économie annuelle potentielle (HT)", f"{economie_cta:,.0f} €/an",
              delta=f"{economie_cta_pct:.1f} %")

# ── Résumé FTA optimale ───────────────────────────────────────────────────────
if optimiser_fta:
    if fta_change:
        st.success(
            f"🔄 **Changement de FTA recommandé** : **{fta}** → **{fta_opt}** "
            f"(économie supplémentaire vs optimisation PS seule : "
            f"{resultat_actuel['Total_HT'] - resultats_fta[fta]['resultat']['Total_HT']:,.0f} € "
            f"→ {economie_cta:,.0f} € avec changement FTA)"
        )
    else:
        st.info(f"✅ La FTA actuelle **{fta}** est déjà la plus avantageuse pour ce profil de consommation.")

    # ── Tableau comparatif FTA ────────────────────────────────────────────────
    st.subheader("📊 Comparaison des formules tarifaires (PS optimisées)")
    rows_fta = []
    for fta_k, v in resultats_fta.items():
        r   = v["resultat"]
        ps  = r["puissances_souscrites"]
        ps_str = " / ".join(f"{p}={int(ps[p])} kVA" for p in ps) if len(ps) > 1 else f"{int(list(ps.values())[0])} kVA"
        marker = " ★" if fta_k == fta_opt else (" ←" if fta_k == fta else "")
        ecart  = round(resultat_actuel["Total_HT"] - r["Total_HT"], 0)
        rows_fta.append({
            "FTA":               fta_k + marker,
            "PS optimisées":     ps_str,
            "TURPE HT":          f"{r['Total']:,.0f} €",
            "CTA HT":            f"{r['CTA_HT']:,.0f} €",
            "Total HT":          f"{r['Total_HT']:,.0f} €",
            "Économie vs actuel": f"{'+' if ecart >= 0 else ''}{ecart:,.0f} €",
            "_total_ht_num":     r["Total_HT"],   # colonne cachée pour le tri
        })
    df_fta = pd.DataFrame(rows_fta).sort_values("_total_ht_num").drop(columns=["_total_ht_num"])

    def style_fta(row):
        tag = str(row["FTA"])
        if "★" in tag:
            return ["background-color: #A5D6A7; color: black; font-weight: bold"] * len(row)
        elif "←" in tag:
            return ["background-color: #90CAF9; color: black"] * len(row)
        return [""] * len(row)

    st.dataframe(df_fta.style.apply(style_fta, axis=1),
                 use_container_width=True, hide_index=True)

st.divider()

# ── Tableau PS ────────────────────────────────────────────────────────────────
st.subheader(f"📋 Puissances souscrites recommandées — FTA optimale : {fta_opt}")

ps_opt_fta_act = resultat_fta_actuelle["puissances_souscrites"]
ps_opt_fta_opt = resultat_optimal["puissances_souscrites"]

if fta_change:
    df_comp = pd.DataFrame({
        "Plage":                          list(ps_actuelles.keys()),
        "Actuelle":                        [f"{ps_actuelles[p]} kVA" for p in ps_actuelles],
        f"PS opt. FTA act. ({fta})":       [f"{ps_opt_fta_act.get(p, 0)} kVA" for p in ps_actuelles],
        f"PS opt. FTA opt. ({fta_opt}) ★": [f"{ps_opt_fta_opt.get(p, 0)} kVA" for p in ps_actuelles],
    })
    _ecarts_opt = [ps_opt_fta_opt.get(p, 0) - ps_actuelles[p] for p in ps_actuelles]

    def style_ecart_ps(row):
        idx = df_comp.index[df_comp["Plage"] == row["Plage"]].tolist()
        if not idx: return [""] * len(row)
        v = _ecarts_opt[idx[0]]
        base = ["", "", ""]
        if v < 0: base.append("color: green; font-weight: bold")
        elif v > 0: base.append("color: red")
        else: base.append("")
        return base

    st.dataframe(df_comp.style.apply(style_ecart_ps, axis=1),
                 use_container_width=True, hide_index=True)
else:
    df_comp = pd.DataFrame({
        "Plage":        list(ps_actuelles.keys()),
        "PS actuelle":  [f"{ps_actuelles[p]} kVA" for p in ps_actuelles],
        "PS optimisée": [f"{ps_opt_fta_opt.get(p, 0)} kVA" for p in ps_actuelles],
        "Écart":        [f"{ps_opt_fta_opt.get(p, 0) - ps_actuelles[p]:+d} kVA" for p in ps_actuelles],
    })
    _ecarts_num = [ps_opt_fta_opt.get(p, 0) - ps_actuelles[p] for p in ps_actuelles]

    def style_ecart_ps(row):
        idx = df_comp.index[df_comp["Plage"] == row["Plage"]].tolist()
        if not idx: return [""] * len(row)
        v = _ecarts_num[idx[0]]
        if v < 0: return ["", "", "", "color: green; font-weight: bold"]
        if v > 0: return ["", "", "", "color: red"]
        return [""] * len(row)

    st.dataframe(df_comp.style.apply(style_ecart_ps, axis=1),
                 use_container_width=True, hide_index=True)

# ── Composantes ───────────────────────────────────────────────────────────────
st.subheader("🔍 Détail des composantes TURPE + CTA — HT")
composantes = ["CG", "CC", "CS", "CMDPS", "CTA_HT"]
labels_comp = {
    "CG":     "Gestion (CG)",
    "CC":     "Comptage (CC)",
    "CS":     "Soutirage (CS)",
    "CMDPS":  "Dépassement (CMDPS)",
    "CTA_HT": "CTA HT (15 %)",
}

if fta_change:
    df_compo_tab = pd.DataFrame({
        "Composante":                    [labels_comp[c] for c in composantes],
        "Actuel":                        [f"{resultat_actuel[c]:,.0f} €/an"       for c in composantes],
        f"PS opt. ({fta})":              [f"{resultat_fta_actuelle[c]:,.0f} €/an"  for c in composantes],
        f"Optimal ({fta_opt}) ★":        [f"{resultat_optimal[c]:,.0f} €/an"       for c in composantes],
        "Écart total":                   [f"{resultat_optimal[c] - resultat_actuel[c]:+,.0f} €/an" for c in composantes],
    })
else:
    df_compo_tab = pd.DataFrame({
        "Composante": [labels_comp[c] for c in composantes],
        "Actuel":     [f"{resultat_actuel[c]:,.0f} €/an"  for c in composantes],
        "Optimisé":   [f"{resultat_optimal[c]:,.0f} €/an" for c in composantes],
        "Écart":      [f"{resultat_optimal[c] - resultat_actuel[c]:+,.0f} €/an" for c in composantes],
    })

col_tab, col_chart = st.columns(2)
with col_tab:
    st.dataframe(df_compo_tab, use_container_width=True, hide_index=True)
with col_chart:
    labels_graph = [labels_comp[c] for c in composantes]
    fig_compo = go.Figure(data=[
        go.Bar(name="Actuel",   x=labels_graph, y=[resultat_actuel[c]          for c in composantes], marker_color="#FF6B6B"),
        go.Bar(name=f"PS opt. ({fta})",   x=labels_graph, y=[resultat_fta_actuelle[c]  for c in composantes], marker_color="#FFB74D"),
        go.Bar(name=f"Optimal ({fta_opt})", x=labels_graph, y=[resultat_optimal[c] for c in composantes], marker_color="#4CAF50"),
    ] if fta_change else [
        go.Bar(name="Actuel",   x=labels_graph, y=[resultat_actuel[c]  for c in composantes], marker_color="#FF6B6B"),
        go.Bar(name="Optimisé", x=labels_graph, y=[resultat_optimal[c] for c in composantes], marker_color="#4CAF50"),
    ])
    fig_compo.update_layout(barmode="group",
                            title="TURPE + CTA HT : actuel vs optimisé",
                            yaxis_title="€/an HT", height=300)
    st.plotly_chart(fig_compo, use_container_width=True)

# ── Courbe avec seuils ────────────────────────────────────────────────────────
st.subheader("📉 Courbe de charge et puissances souscrites optimisées")
fig_final = go.Figure()
fig_final.add_trace(go.Scatter(
    x=df["timestamp"], y=df["puissance_kw"],
    mode="lines", name="Consommation", line=dict(color="#2196F3", width=1),
))
for plage, ps in resultat_optimal["puissances_souscrites"].items():
    fig_final.add_hline(y=ps, line_dash="dot", line_color=COULEURS_PLAGES.get(plage, "#888"),
                        annotation_text=f"PS {plage} : {ps} kVA", annotation_position="right")
fig_final.update_layout(title="Courbe de charge et seuils optimisés",
                        xaxis_title="Date", yaxis_title="kW / kVA", height=380)
st.plotly_chart(fig_final, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# SENSIBILITÉ — avec option superposition
# ─────────────────────────────────────────────
st.header("📈 Analyse de sensibilité")

if "plage_variee" in df_scenarios.columns:
    mode_superpose = st.checkbox("Superposer toutes les plages sur un même graphique", value=True)

    if mode_superpose:
        fig_sens = go.Figure()
        for plage in df_scenarios["plage_variee"].unique():
            df_s = df_scenarios[df_scenarios["plage_variee"] == plage].sort_values("ps_variee")
            fig_sens.add_trace(go.Scatter(
                x=df_s["ps_variee"], y=df_s["Total"],
                mode="lines+markers",
                name=plage,
                line=dict(color=COULEURS_PLAGES.get(plage, "#888"), width=2),
                marker=dict(size=4),
            ))
            ps_opt_plage = resultat_optimal["puissances_souscrites"].get(plage, 0)
            fig_sens.add_vline(
                x=ps_opt_plage, line_dash="dot",
                line_color=COULEURS_PLAGES.get(plage, "#888"),
                annotation_text=f"Opt. {plage}: {ps_opt_plage}",
                annotation_font_size=9,
            )
        fig_sens.update_layout(
            title="Sensibilité du coût TURPE HT — toutes plages superposées",
            xaxis_title="Puissance souscrite (kVA)",
            yaxis_title="Coût TURPE HT annualisé (€/an)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_sens, use_container_width=True)
    else:
        plage_sel = st.selectbox("Plage à analyser", df_scenarios["plage_variee"].unique())
        df_sens   = df_scenarios[df_scenarios["plage_variee"] == plage_sel].sort_values("ps_variee")
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=df_sens["ps_variee"], y=df_sens["Total"],
            mode="lines+markers",
            line=dict(color=COULEURS_PLAGES.get(plage_sel, "#2196F3"), width=2),
            name="Coût total (€/an)",
        ))
        fig_sens.add_vline(x=resultat_optimal["puissances_souscrites"].get(plage_sel, 0),
                           line_dash="dash", line_color="#4CAF50", annotation_text="✅ Optimal")
        fig_sens.add_vline(x=ps_actuelles.get(plage_sel, 0),
                           line_dash="dash", line_color="#FF4444", annotation_text="📌 Actuel")
        fig_sens.update_layout(
            title=f"Sensibilité du coût — {plage_sel}",
            xaxis_title="Puissance souscrite (kVA)", yaxis_title="€/an", height=340,
        )
        st.plotly_chart(fig_sens, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# PROJECTION PLURIANNUELLE
# ─────────────────────────────────────────────
st.header("📆 Gains annuels et projection")

col_proj1, col_proj2 = st.columns([1, 3])
with col_proj1:
    nb_annees        = st.slider("Horizon (années)", 1, 20, 10)
    eco_cta_annuelle = max(0, economie_cta)

with col_proj2:
    annees           = list(range(1, nb_annees + 1))
    eco_cta_annuelle = max(0, economie_cta)
    eco_cumul        = [eco_cta_annuelle * a for a in annees]

    fig_projection = go.Figure()
    fig_projection.add_trace(go.Bar(
        x=annees, y=[eco_cta_annuelle] * nb_annees,
        name="Économie annuelle HT",
        marker_color="#4CAF50", opacity=0.75,
    ))
    fig_projection.add_trace(go.Scatter(
        x=annees, y=eco_cumul,
        mode="lines+markers", name="Cumul",
        line=dict(color="#1565C0", width=2),
        marker=dict(size=5),
        yaxis="y2",
    ))
    fig_projection.update_layout(
        title=f"Projection des gains sur {nb_annees} ans — FTA {fta_opt}",
        xaxis=dict(title="Année", tickmode="linear", dtick=1 if nb_annees <= 15 else 2),
        yaxis=dict(title="Économie annuelle (€ HT)", side="left", showgrid=True),
        yaxis2=dict(title="Cumul (€ HT)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
    )
    st.plotly_chart(fig_projection, use_container_width=True)

    if eco_cta_annuelle > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Économie annuelle HT",       f"{eco_cta_annuelle:,.0f} €/an")
        c2.metric(f"Cumul sur {nb_annees} ans", f"{eco_cta_annuelle * nb_annees:,.0f} €")
        c3.metric("Gain relatif",               f"{economie_cta_pct:.1f} %")

st.divider()

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
st.header("💾 Export des résultats")

col_dl1, col_dl2, col_dl3 = st.columns(3)

# CSV synthèse
rapport = pd.DataFrame({
    "Paramètre": [
        "Nom de l'étude", "Format fichier", "Domaine", "FTA", "Contrat",
        "HC début", "HC fin",
        "PRM", "Période", "Durée (jours)", "Couverture annuelle",
        "Coût TURPE+CTA actuel HT (€/an)", "Coût TURPE+CTA optimisé HT (€/an)",
        "Économie HT (€/an)", "Économie (%)",
    ],
    "Valeur": [
        nom_etude,
        st.session_state.get("format_detecte", "?"), domaine, fta, type_contrat,
        f"{hc_debut}h", f"{hc_fin}h",
        df_raw.attrs.get("prm", "?"),
        f"{df_raw.attrs.get('periode_debut','?')} → {df_raw.attrs.get('periode_fin','?')}",
        df_raw.attrs.get("nb_jours", "?"),
        f"{round(df_raw.attrs.get('nb_jours', 365)/365*100, 1)} %",
        f"{resultat_actuel['Total_HT']:,.0f} €",
        f"{resultat_optimal['Total_HT']:,.0f} €",
        f"{economie_cta:,.0f} €", f"{economie_cta_pct:.1f} %",
    ]
})
for plage in ps_actuelles:
    rapport = pd.concat([rapport, pd.DataFrame({
        "Paramètre": [f"PS actuelle {plage}", f"PS optimisée {plage}"],
        "Valeur": [f"{ps_actuelles[plage]} kVA",
                   f"{resultat_optimal['puissances_souscrites'].get(plage,'?')} kVA"],
    })], ignore_index=True)

with col_dl1:
    st.download_button("📥 Rapport synthèse (CSV)",
        data=rapport.to_csv(index=False, sep=";"),
        file_name=f"rapport_turpe7_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv")

with col_dl2:
    st.download_button("📥 Courbe classée (CSV)",
        data=df[["timestamp", "puissance_kw", "plage"]].to_csv(index=False, sep=";"),
        file_name=f"courbe_classee_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv")

with col_dl3:
    if st.button("📄 Générer le rapport PDF", type="primary"):
        with st.spinner("Génération du PDF en cours..."):
            try:
                pdf_bytes = generer_pdf(
                    df_raw=df_raw, df=df_opt,
                    nom_etude=nom_etude,
                    domaine=domaine, fta=fta, fta_opt=fta_opt, type_contrat=type_contrat,
                    hc_debut=hc_debut, hc_fin=hc_fin,
                    ps_actuelles=ps_actuelles,
                    resultat_actuel=resultat_actuel,
                    resultat_optimal=resultat_optimal,
                    economie=economie, economie_pct=economie_pct,
                    economie_cta=economie_cta, economie_cta_pct=economie_cta_pct,
                    nb_annees=nb_annees,
                    resultats_fta=resultats_fta,
                )
                st.session_state["pdf_bytes"] = pdf_bytes
                st.success("✅ PDF généré !")
            except Exception as e:
                st.error(f"❌ Erreur PDF : {e}")

    if "pdf_bytes" in st.session_state:
        nom_fichier = nom_etude.strip().replace(" ", "_").replace("/", "-") or "rapport"
        st.download_button(
            "⬇️ Télécharger le PDF",
            data=st.session_state["pdf_bytes"],
            file_name=f"{nom_fichier}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )

st.divider()
st.caption("📌 TURPE 7 + CTA — Réseau Enedis — Tarifs au 1er février 2026 — Montants HT — Contrainte HPH ≤ HCH ≤ HPB ≤ HCB")
