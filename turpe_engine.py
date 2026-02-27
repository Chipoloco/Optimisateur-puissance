"""
Moteur de calcul TURPE 7 — Enedis
En vigueur au 1er août 2025 (Délibération CRE n°2025-78 du 13 mars 2025)

Formats supportés :
- Enedis SGE R63  : colonnes Horodate / Valeur(W) / séparateur ;
- Likewatt        : colonnes Date de la mesure / Heure de la mesure / PA(W) / séparateur ;

Contrainte TURPE sur les puissances souscrites (BT > 36 kVA et HTA) :
  P_HPH <= P_HCH <= P_HPB <= P_HCB   (et Pointe <= HPH pour HTA)
  → plage la plus chère = PS la plus basse
  → plage la moins chère = PS la plus haute

Heures creuses configurables :
  Par défaut 8h-20h en HP (= 12h HP + 12h HC).
  L'utilisateur peut définir ses propres plages HP via hp_debut / hp_fin.
  HC = tout ce qui n'est pas HP (jours ouvrés), week-end toujours HC.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────────────────────
# 1. CONSTANTES TARIFAIRES TURPE 7
# ─────────────────────────────────────────────

PLAGES_HTA    = ["Pointe", "HPH", "HCH", "HPB", "HCB"]
PLAGES_BT_SUP = ["HPH", "HCH", "HPB", "HCB"]

# HTA : bi (€/kW/an)
HTA_BI = {
    "CU pointe fixe":   {"Pointe": 14.41, "HPH": 14.41, "HCH": 14.41, "HPB": 12.55, "HCB": 11.22},
    "CU pointe mobile": {"Pointe": 14.41, "HPH": 14.41, "HCH": 14.41, "HPB": 12.55, "HCB": 11.22},
    "LU pointe fixe":   {"Pointe": 35.33, "HPH": 32.30, "HCH": 20.39, "HPB": 14.33, "HCB": 11.56},
    "LU pointe mobile": {"Pointe": 38.27, "HPH": 34.30, "HCH": 20.39, "HPB": 14.33, "HCB": 11.56},
}

# HTA : ci (c€/kWh)
HTA_CI = {
    "CU pointe fixe":   {"Pointe": 5.74, "HPH": 4.23, "HCH": 1.99, "HPB": 1.01, "HCB": 0.69},
    "CU pointe mobile": {"Pointe": 7.01, "HPH": 4.05, "HCH": 1.99, "HPB": 1.01, "HCB": 0.69},
    "LU pointe fixe":   {"Pointe": 2.65, "HPH": 2.10, "HCH": 1.47, "HPB": 0.92, "HCB": 0.68},
    "LU pointe mobile": {"Pointe": 3.15, "HPH": 1.87, "HCH": 1.47, "HPB": 0.92, "HCB": 0.68},
}

# BT > 36 kVA : bi (€/kVA/an)
BT_SUP_BI = {
    "CU": {"HPH": 17.61, "HCH": 15.96, "HPB": 14.56, "HCB": 11.98},
    "LU": {"HPH": 30.16, "HCH": 21.18, "HPB": 16.64, "HCB": 12.37},
}

# BT > 36 kVA : ci (c€/kWh)
BT_SUP_CI = {
    "CU": {"HPH": 6.91, "HCH": 4.21, "HPB": 2.13, "HCB": 1.52},
    "LU": {"HPH": 5.69, "HCH": 3.47, "HPB": 2.01, "HCB": 1.49},
}

# BT ≤ 36 kVA : b unique (€/kVA/an)
BT_INF_B = {
    "CU4":                10.11,
    "MU4":                12.12,
    "LU":                 93.13,
    "CU (dérogatoire)":   11.07,
    "MUDT (dérogatoire)": 13.49,
}

# BT ≤ 36 kVA : ci (c€/kWh)
BT_INF_CI = {
    "CU4":  {"HPH": 7.49, "HCH": 3.97, "HPB": 1.66, "HCB": 1.16},
    "MU4":  {"HPH": 7.00, "HCH": 3.73, "HPB": 1.61, "HCB": 1.11},
    "LU":   {"HPH": 1.25, "HCH": 1.25, "HPB": 1.25, "HCB": 1.25},
    "CU (dérogatoire)":   {"HPH": 4.84, "HCH": 4.84, "HPB": 4.84, "HCB": 4.84},
    "MUDT (dérogatoire)": {"HPH": 4.94, "HCH": 3.50, "HPB": 4.94, "HCB": 3.50},
}

# Composantes fixes (€/an)
COMPOSANTES_FIXES = {
    "HTA":    {"CG_contrat_unique": 435.72, "CG_CARD": 499.80, "CC": 376.39},
    "BT_SUP": {"CG_contrat_unique": 217.80, "CG_CARD": 249.84, "CC": 283.27},
    "BT_INF": {"CG_contrat_unique":  16.80, "CG_CARD":  18.00, "CC":  22.00},
}


# ─────────────────────────────────────────────
# 2. INGESTION — FONCTIONS COMMUNES
# ─────────────────────────────────────────────

def _finaliser_dataframe(df_clean: pd.DataFrame, resolution_source: str) -> pd.DataFrame:
    """
    Prend un DataFrame propre (timestamp, puissance_kw) et produit
    le DataFrame horaire final avec puissance_kw_10min et attrs d'annualisation.
    """
    df_10min = (
        df_clean.set_index("timestamp")["puissance_kw"]
        .resample("10min").max()
        .reset_index()
        .rename(columns={"puissance_kw": "puissance_kw_10min"})
    )

    df_1h = (
        df_clean.set_index("timestamp")["puissance_kw"]
        .resample("h").max()
        .reset_index()
    )

    df_1h = df_1h.merge(
        df_10min
        .assign(timestamp=df_10min["timestamp"].dt.floor("h"))
        .groupby("timestamp")["puissance_kw_10min"].max()
        .reset_index(),
        on="timestamp", how="left"
    )

    debut    = df_clean["timestamp"].min()
    fin      = df_clean["timestamp"].max()
    nb_jours = max(1, (fin - debut).days)
    fact_ann = round(365.0 / nb_jours, 4)

    df_1h.attrs["prm"]                   = df_clean.get("prm", pd.Series([None])).iloc[0] if "prm" in df_clean.columns else None
    df_1h.attrs["periode_debut"]         = debut
    df_1h.attrs["periode_fin"]           = fin
    df_1h.attrs["nb_jours"]              = nb_jours
    df_1h.attrs["facteur_annualisation"] = fact_ann
    df_1h.attrs["resolution_source"]     = resolution_source

    return df_1h


# ─────────────────────────────────────────────
# 2a. FORMAT ENEDIS SGE R63
# ─────────────────────────────────────────────

def charger_fichier_enedis(filepath_or_buffer) -> pd.DataFrame:
    """
    Charge un fichier CSV Enedis SGE format R63.
    Colonnes : Horodate (datetime) | Valeur (W) | Unité | Identifiant PRM
    Séparateur : ;  |  Pas : PT5M (5 min)
    """
    df = pd.read_csv(filepath_or_buffer, sep=";", parse_dates=["Horodate"], low_memory=False)

    colonnes_requises = {"Horodate", "Valeur", "Unité"}
    if not colonnes_requises.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes pour format R63. Trouvées : {set(df.columns)}")

    if not (df["Unité"] == "W").all():
        raise ValueError(f"Unité inattendue : {df['Unité'].unique()}. Seul 'W' supporté.")

    df_clean = pd.DataFrame({
        "timestamp":    df["Horodate"],
        "puissance_kw": df["Valeur"] / 1000.0,
        "prm":          df["Identifiant PRM"].iloc[0] if "Identifiant PRM" in df.columns else None,
    }).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    resolution = df["Pas"].iloc[0] if "Pas" in df.columns else "PT5M"
    return _finaliser_dataframe(df_clean, resolution)


# ─────────────────────────────────────────────
# 2b. FORMAT LIKEWATT
# ─────────────────────────────────────────────

def charger_fichier_likewatt(filepath_or_buffer) -> pd.DataFrame:
    """
    Charge un fichier CSV format Likewatt (export courbe de charge).
    Colonnes : Date de la mesure (DD-MM-YYYY) | Heure de la mesure (HH:MM) | PA (W)
    Séparateur : ;  |  Pas : 5 min
    """
    df = pd.read_csv(filepath_or_buffer, sep=";", low_memory=False)

    colonnes_requises = {"Date de la mesure", "Heure de la mesure", "PA"}
    if not colonnes_requises.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes pour format Likewatt. Trouvées : {set(df.columns)}")

    # Fusion date + heure → timestamp (format DD-MM-YYYY HH:MM)
    df["timestamp"] = pd.to_datetime(
        df["Date de la mesure"].astype(str) + " " + df["Heure de la mesure"].astype(str),
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    df_clean = pd.DataFrame({
        "timestamp":    df["timestamp"],
        "puissance_kw": pd.to_numeric(df["PA"], errors="coerce") / 1000.0,
        "prm":          df["PRM"].iloc[0] if "PRM" in df.columns else None,
    }).dropna(subset=["timestamp", "puissance_kw"])\
      .sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    pas = df["Pas en minutes"].dropna().iloc[0] if "Pas en minutes" in df.columns else 5
    return _finaliser_dataframe(df_clean, f"PT{int(pas)}M")


# ─────────────────────────────────────────────
# 2c. DÉTECTION AUTOMATIQUE DU FORMAT
# ─────────────────────────────────────────────

def charger_fichier_auto(filepath_or_buffer) -> Tuple[pd.DataFrame, str]:
    """
    Détecte automatiquement le format (R63 ou Likewatt) et charge le fichier.
    Retourne (dataframe, format_detecte).
    """
    import io

    # Lecture de l'en-tête pour détection
    if hasattr(filepath_or_buffer, "read"):
        contenu = filepath_or_buffer.read()
        if isinstance(contenu, bytes):
            contenu = contenu.decode("utf-8", errors="replace")
        premiere_ligne = contenu.split("\n")[0]
        buffer = io.StringIO(contenu)
    else:
        with open(filepath_or_buffer, "r", encoding="utf-8", errors="replace") as f:
            premiere_ligne = f.readline()
        buffer = filepath_or_buffer

    if "Horodate" in premiere_ligne:
        return charger_fichier_enedis(buffer), "Enedis R63"
    elif "Date de la mesure" in premiere_ligne:
        return charger_fichier_likewatt(buffer), "Likewatt"
    else:
        raise ValueError(
            "Format de fichier non reconnu. "
            "Formats supportés : Enedis SGE R63 (colonne 'Horodate') "
            "ou Likewatt (colonnes 'Date de la mesure' / 'Heure de la mesure' / 'PA')."
        )


def resumer_chargement(df: pd.DataFrame) -> dict:
    """Retourne un résumé lisible du DataFrame chargé."""
    nb_jours = df.attrs.get("nb_jours", 365)
    fact     = df.attrs.get("facteur_annualisation", 1.0)
    couv_pct = round(nb_jours / 365 * 100, 1)
    return {
        "PRM":                     df.attrs.get("prm", "inconnu"),
        "Période":                 f"{df.attrs.get('periode_debut', '?')} → {df.attrs.get('periode_fin', '?')}",
        "Nombre de jours":         nb_jours,
        "Couverture annuelle":     f"{couv_pct} %",
        "Facteur d'annualisation": f"×{fact}",
        "Résolution source":       df.attrs.get("resolution_source", "?"),
        "Points horaires":         len(df),
        "Puissance min (kW)":      round(df["puissance_kw"].min(), 1),
        "Puissance moyenne (kW)":  round(df["puissance_kw"].mean(), 1),
        "Puissance max (kW)":      round(df["puissance_kw"].max(), 1),
        "Valeurs manquantes":      int(df["puissance_kw"].isna().sum()),
    }


# ─────────────────────────────────────────────
# 3. CLASSIFICATION DES HEURES
# ─────────────────────────────────────────────

def classifier_plage(
    timestamp: pd.Timestamp,
    domaine: str,
    fta: str,
    hc_debut: int = 22,
    hc_fin: int = 6,
) -> str:
    """
    Retourne la plage temporelle d'un timestamp.

    Paramètres :
    - hc_debut / hc_fin : bornes des Heures Creuses (défaut 22h → 6h).
      Si hc_debut > hc_fin : les HC chevauchent minuit (ex: 22h→6h).
      Si hc_debut < hc_fin : les HC sont en journée (ex: 0h→8h).
      HC s'appliquent en jours ouvrés (lun-ven).
      Week-end (sam-dim) : toujours HC.

    Saison haute : novembre à mars inclus.
    Pointe HTA fixe : 8h-10h et 17h-19h, déc-fév, hors dimanche.
    """
    mois         = timestamp.month
    heure        = timestamp.hour
    jour_semaine = timestamp.weekday()   # 0=lundi … 6=dimanche
    saison_haute = mois in [11, 12, 1, 2, 3]

    # Calcul est_hc selon que les HC chevauchent minuit ou non (valable 7j/7)
    if hc_debut > hc_fin:
        est_hc = (heure >= hc_debut) or (heure < hc_fin)
    else:
        est_hc = (hc_debut <= heure < hc_fin)

    est_hp = not est_hc

    if domaine == "HTA":
        if "pointe fixe" in fta:
            if mois in [12, 1, 2] and jour_semaine < 6:
                if 8 <= heure < 10 or 17 <= heure < 19:
                    return "Pointe"
        elif "pointe mobile" in fta:
            if mois in [12, 1, 2] and jour_semaine < 6:
                if 7 <= heure < 15 or 18 <= heure < 20:
                    return "Pointe"
        return ("HPH" if est_hp else "HCH") if saison_haute \
          else ("HPB" if est_hp else "HCB")
    else:
        return ("HPH" if est_hp else "HCH") if saison_haute \
          else ("HPB" if est_hp else "HCB")


def classifier_dataframe(
    df: pd.DataFrame,
    domaine: str,
    fta: str,
    hc_debut: int = 22,
    hc_fin: int = 6,
) -> pd.DataFrame:
    """Ajoute une colonne 'plage' au DataFrame en préservant ses attrs."""
    attrs_backup = df.attrs.copy()
    df = df.copy()
    df["plage"] = df["timestamp"].apply(
        lambda t: classifier_plage(t, domaine, fta, hc_debut, hc_fin)
    )
    df.attrs = attrs_backup
    return df


# ─────────────────────────────────────────────
# 4. CALCUL DU COÛT TURPE ANNUALISÉ
# ─────────────────────────────────────────────

def calculer_cout_total(
    df: pd.DataFrame,
    domaine: str,
    fta: str,
    puissances_souscrites: Dict[str, float],
    type_contrat: str = "contrat_unique",
) -> Dict:
    """
    Calcule le coût TURPE annuel (€/an).
    Annualisation automatique via facteur_annualisation dans les attrs.

    Contrainte respectée : P_HPH <= P_HCH <= P_HPB <= P_HCB
    (la plage la plus chère a la PS la plus basse)
    """
    domaine_key = {"HTA": "HTA", "BT > 36 kVA": "BT_SUP", "BT ≤ 36 kVA": "BT_INF"}[domaine]
    fixes    = COMPOSANTES_FIXES[domaine_key]
    cg       = fixes[f"CG_{type_contrat}"]
    cc       = fixes["CC"]
    fact_ann = df.attrs.get("facteur_annualisation", 1.0)

    energies = df.groupby("plage")["puissance_kw"].sum().to_dict()

    if domaine == "HTA":
        bi = HTA_BI[fta]
        ci = HTA_CI[fta]

        # Formule dégressive : tri par bi DÉCROISSANT (plage la plus chère en 1er)
        # Contrainte : P[i] <= P[i+1] (PS croissante quand bi décroît)
        ordre    = sorted(PLAGES_HTA, key=lambda p: bi[p], reverse=True)
        ps_tries = [puissances_souscrites[p] for p in ordre]

        cs_puissance = bi[ordre[0]] * ps_tries[0]
        for idx in range(1, len(ordre)):
            cs_puissance += bi[ordre[idx]] * (ps_tries[idx] - ps_tries[idx - 1])
        cs_energie = sum((ci[p] / 100) * energies.get(p, 0) * fact_ann for p in PLAGES_HTA)
        cs = cs_puissance + cs_energie

        col_p = "puissance_kw_10min" if "puissance_kw_10min" in df.columns else "puissance_kw"
        cmdps = 0.0
        for (_, plage), groupe in df.groupby([df["timestamp"].dt.month, "plage"]):
            ps     = puissances_souscrites.get(plage, 0)
            deltas = np.maximum(0, groupe[col_p].values - ps)
            if deltas.sum() > 0:
                cmdps += 0.04 * bi[plage] * np.sqrt(np.sum(deltas ** 2))
        cmdps *= fact_ann

    elif domaine == "BT > 36 kVA":
        bi = BT_SUP_BI[fta]
        ci = BT_SUP_CI[fta]

        # Même logique : tri bi DÉCROISSANT → HPH(17.61) > HCH(15.96) > HPB(14.56) > HCB(11.98)
        # → P_HPH <= P_HCH <= P_HPB <= P_HCB
        ordre    = sorted(PLAGES_BT_SUP, key=lambda p: bi[p], reverse=True)
        ps_tries = [puissances_souscrites[p] for p in ordre]

        cs_puissance = bi[ordre[0]] * ps_tries[0]
        for idx in range(1, len(ordre)):
            cs_puissance += bi[ordre[idx]] * (ps_tries[idx] - ps_tries[idx - 1])
        cs_energie = sum((ci[p] / 100) * energies.get(p, 0) * fact_ann for p in PLAGES_BT_SUP)
        cs = cs_puissance + cs_energie

        heures_dep = sum(
            (df[df["plage"] == p]["puissance_kw"] > ps).sum()
            for p, ps in puissances_souscrites.items()
        )
        cmdps = 12.41 * heures_dep * fact_ann

    else:  # BT ≤ 36 kVA
        ps_unique = list(puissances_souscrites.values())[0]
        b  = BT_INF_B[fta]
        ci = BT_INF_CI[fta]
        cs_puissance = b * ps_unique
        cs_energie   = sum((ci[p] / 100) * energies.get(p, 0) * fact_ann for p in ci)
        cs    = cs_puissance + cs_energie
        cmdps = 0.0

    # ── CTA (Contribution Tarifaire d'Acheminement) ───────────────────────────
    # Taux : 15 % depuis le 1er février 2026 pour les clients Enedis (distribution)
    # Assiette : CG + CC + CS_puissance (part fixe uniquement, hors énergie et CMDPS)
    # TVA : 20 % sur la CTA (taux pro depuis le 1er août 2025)
    TAUX_CTA     = 0.15
    TAUX_TVA_CTA = 0.20
    cta_ht       = (cg + cc + cs_puissance) * TAUX_CTA
    cta_ttc      = cta_ht * (1 + TAUX_TVA_CTA)

    total         = cg + cc + cs + cmdps
    total_avec_cta = total + cta_ttc

    return {
        "CG":               round(cg, 2),
        "CC":               round(cc, 2),
        "CS":               round(cs, 2),
        "CS_puissance":     round(cs_puissance, 2),
        "CMDPS":            round(cmdps, 2),
        "CTA_HT":           round(cta_ht, 2),
        "CTA_TTC":          round(cta_ttc, 2),
        "Total":            round(total, 2),
        "Total_avec_CTA":   round(total_avec_cta, 2),
        "facteur_annualisation": fact_ann,
        "puissances_souscrites": puissances_souscrites,
    }


# ─────────────────────────────────────────────
# 5. MOTEUR D'OPTIMISATION
# ─────────────────────────────────────────────

def _appliquer_contrainte(ps: Dict[str, float], bi: Dict[str, float]) -> Dict[str, float]:
    """
    Applique la contrainte TURPE : P_HPH <= P_HCH <= P_HPB <= P_HCB.
    Tri par bi décroissant (plus cher = PS la plus basse).
    Chaque plage moins chère reçoit au minimum la PS de la plus chère précédente.
    """
    ps = dict(ps)
    plages_dec = sorted(ps.keys(), key=lambda p: bi[p], reverse=True)
    ps_min = 0
    for p in plages_dec:
        ps[p] = max(ps[p], ps_min)
        ps_min = ps[p]
    return ps


def optimiser_puissances(
    df: pd.DataFrame,
    domaine: str,
    fta: str,
    type_contrat: str = "contrat_unique",
    pas_kva: int = 1,
    ps_actuelles: Optional[Dict[str, float]] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Optimise les puissances souscrites par plage.

    Algorithme : descente par coordonnées avec balayage global initial.
    1. Balayage d'une PS commune à toutes les plages → point de départ.
    2. Raffinement par coordonnées : optimise chaque plage en fixant les autres
       à leurs meilleures valeurs courantes. Répété jusqu'à convergence.
    3. Application de la contrainte HPH <= HCH <= HPB <= HCB.
    4. Non-régression : si le résultat est moins bon que la situation actuelle,
       retourne la situation actuelle inchangée.

    Retourne :
    - Dict : PS optimales + coût annualisé
    - DataFrame : scénarios de sensibilité autour de l'optimal
    """
    plages = (
        PLAGES_HTA     if domaine == "HTA"         else
        PLAGES_BT_SUP  if domaine == "BT > 36 kVA" else
        ["unique"]
    )

    if domaine in ["HTA", "BT > 36 kVA"]:
        bi = HTA_BI[fta] if domaine == "HTA" else BT_SUP_BI[fta]

        p_global_max  = df["puissance_kw"].max()
        max_par_plage = {
            p: (df[df["plage"] == p]["puissance_kw"].max() if (df["plage"] == p).any() else p_global_max)
            for p in plages
        }
        p_global_min = max(1, df["puissance_kw"].quantile(0.40))

        # ── Étape 1 : balayage global (même PS pour toutes les plages) ─────────
        # Trouve un bon point de départ sans biais inter-plage
        meilleur_cout_global = float("inf")
        ps_commune_opt = int(p_global_max)

        borne_min = max(1, int(p_global_min) - pas_kva)
        borne_max = int(p_global_max) + pas_kva * 2

        for ps_val in range(borne_min, borne_max, pas_kva):
            ps_test = {p: ps_val for p in plages}
            r = calculer_cout_total(df.copy(), domaine, fta, ps_test, type_contrat)
            if r["Total"] < meilleur_cout_global:
                meilleur_cout_global = r["Total"]
                ps_commune_opt = ps_val

        # ── Étape 2 : descente par coordonnées autour du point de départ ───────
        # Chaque plage est optimisée en tenant les autres fixes (itéré 3 fois)
        current_ps = {p: ps_commune_opt for p in plages}
        fenetre = max(20, pas_kva * 30)  # fenêtre de recherche autour du point de départ

        for _ in range(3):  # 3 itérations suffisent pour la convergence
            for plage in plages:
                meilleur_cout_plage = float("inf")
                meilleure_ps_plage  = current_ps[plage]
                borne_inf = max(1, current_ps[plage] - fenetre)
                borne_sup = min(int(max_par_plage[plage]) + pas_kva, current_ps[plage] + fenetre)

                for ps_cand in range(borne_inf, borne_sup + pas_kva, pas_kva):
                    ps_test = dict(current_ps)
                    ps_test[plage] = ps_cand
                    r = calculer_cout_total(df.copy(), domaine, fta, ps_test, type_contrat)
                    if r["Total"] < meilleur_cout_plage:
                        meilleur_cout_plage = r["Total"]
                        meilleure_ps_plage  = ps_cand
                current_ps[plage] = meilleure_ps_plage

        # ── Étape 3 : application de la contrainte ────────────────────────────
        meilleures_ps = _appliquer_contrainte(current_ps, bi)

        # ── Étape 4 : non-régression ──────────────────────────────────────────
        # Si on a la situation actuelle en entrée, ne pas proposer pire
        resultat_optimal = calculer_cout_total(df.copy(), domaine, fta, meilleures_ps, type_contrat)
        if ps_actuelles is not None:
            resultat_ref = calculer_cout_total(df.copy(), domaine, fta, ps_actuelles, type_contrat)
            if resultat_optimal["Total"] >= resultat_ref["Total"]:
                meilleures_ps    = dict(ps_actuelles)
                resultat_optimal = resultat_ref

        # ── Scénarios de sensibilité autour de l'optimal ─────────────────────
        scenarios = []
        for plage in plages:
            ps_opt = meilleures_ps[plage]
            for delta in range(-5 * pas_kva, 6 * pas_kva, pas_kva):
                ps_var = dict(meilleures_ps)
                ps_var[plage] = max(1, ps_opt + delta)
                r = calculer_cout_total(df.copy(), domaine, fta, ps_var, type_contrat)
                r["plage_variee"] = plage
                r["ps_variee"]    = ps_var[plage]
                scenarios.append(r)
        df_scenarios = pd.DataFrame(scenarios)

    else:  # BT ≤ 36 kVA
        p_max         = df["puissance_kw"].max()
        p_min         = max(1, df["puissance_kw"].quantile(0.40))
        scenarios     = []
        meilleur_cout = float("inf")
        meilleure_ps  = p_max

        for ps in range(max(1, int(p_min) - pas_kva), int(p_max) + pas_kva * 2, pas_kva):
            r = calculer_cout_total(df.copy(), domaine, fta, {"unique": ps}, type_contrat)
            r["ps_variee"] = ps
            scenarios.append(r)
            if r["Total"] < meilleur_cout:
                meilleur_cout = r["Total"]
                meilleure_ps  = ps

        meilleures_ps    = {"unique": meilleure_ps}
        resultat_optimal = calculer_cout_total(df.copy(), domaine, fta, meilleures_ps, type_contrat)

        # Non-régression
        if ps_actuelles is not None:
            resultat_ref = calculer_cout_total(df.copy(), domaine, fta, ps_actuelles, type_contrat)
            if resultat_optimal["Total"] >= resultat_ref["Total"]:
                meilleures_ps    = dict(ps_actuelles)
                resultat_optimal = resultat_ref

        df_scenarios = pd.DataFrame(scenarios)

    return resultat_optimal, df_scenarios
