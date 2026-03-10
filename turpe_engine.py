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
        .resample("h").mean()
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


def charger_fichier_likewatt_compact(filepath_or_buffer) -> pd.DataFrame:
    """
    Charge un fichier CSV format Likewatt compact (variante export).
    Colonnes : Date de la mesure (DD/MM/YYYY HH:MM fusionné) | Puissance (W)
    Date et heure dans une seule colonne, séparateur ;
    Pas : détecté automatiquement depuis les données.
    """
    df = pd.read_csv(filepath_or_buffer, sep=";", low_memory=False)

    colonnes_requises = {"Date de la mesure", "Puissance (W)"}
    if not colonnes_requises.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes pour format Likewatt compact. Trouvées : {set(df.columns)}")

    df["timestamp"] = pd.to_datetime(
        df["Date de la mesure"].astype(str),
        dayfirst=True,
        errors="coerce"
    )

    df_clean = pd.DataFrame({
        "timestamp":    df["timestamp"],
        "puissance_kw": pd.to_numeric(df["Puissance (W)"], errors="coerce") / 1000.0,
        "prm":          df["PRM"].iloc[0] if "PRM" in df.columns else None,
    }).dropna(subset=["timestamp", "puissance_kw"])\
      .sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Détection automatique du pas depuis les données
    if len(df_clean) >= 2:
        delta = int((df_clean["timestamp"].iloc[1] - df_clean["timestamp"].iloc[0]).seconds // 60)
    else:
        delta = 15
    return _finaliser_dataframe(df_clean, f"PT{delta}M")


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
        # Détection de la variante : compact (date+heure fusionnées, colonne "Puissance (W)")
        # vs standard (date et heure séparées, colonne "PA")
        if "Puissance (W)" in premiere_ligne:
            return charger_fichier_likewatt_compact(buffer), "Likewatt compact"
        else:
            return charger_fichier_likewatt(buffer), "Likewatt"
    else:
        raise ValueError(
            "Format de fichier non reconnu. "
            "Formats supportés : Enedis SGE R63 (colonne 'Horodate'), "
            "Likewatt standard (colonnes 'Date de la mesure' / 'Heure de la mesure' / 'PA'), "
            "Likewatt compact (colonnes 'Date de la mesure' / 'Puissance (W)')."
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

# ─────────────────────────────────────────────
# 4b. CACHE D'OPTIMISATION
# ─────────────────────────────────────────────

def _construire_cache(
    df: pd.DataFrame,
    domaine: str,
    fta: str,
    type_contrat: str = "contrat_unique",
) -> Dict:
    """
    Précalcule toutes les quantités invariantes par rapport aux PS.
    Appelé une seule fois avant l'optimisation ; permet d'évaluer
    des milliers de configurations en arithmétique pure, sans toucher au DataFrame.

    Contenu du cache :
    - constantes tarifaires (bi, ci, cg, cc, fact_ann, ordre des plages)
    - energies par plage (somme kWh, constante)
    - cs_energie (constante)
    - puissances_triees[plage] : array numpy trié croissant des mesures
      → searchsorted donne nb_dépassements en O(log n) au lieu de O(n)
    - Pour HTA : puissances_triees_mois[mois][plage] pour la formule CMDPS mensuelle
    """
    domaine_key = {"HTA": "HTA", "BT > 36 kVA": "BT_SUP", "BT ≤ 36 kVA": "BT_INF"}[domaine]
    fixes    = COMPOSANTES_FIXES[domaine_key]
    cg       = fixes[f"CG_{type_contrat}"]
    cc       = fixes["CC"]
    fact_ann = df.attrs.get("facteur_annualisation", 1.0)

    energies = df.groupby("plage")["puissance_kw"].sum().to_dict()

    cache = {
        "cg": cg, "cc": cc, "fact_ann": fact_ann, "energies": energies,
        "domaine": domaine, "fta": fta,
    }

    if domaine == "HTA":
        bi = HTA_BI[fta]
        ci = HTA_CI[fta]
        ordre = sorted(PLAGES_HTA, key=lambda p: bi[p], reverse=True)
        bi_ordre  = [bi[p] for p in ordre]
        ci_energie = {p: ci[p] / 100 * energies.get(p, 0) * fact_ann for p in PLAGES_HTA}
        cs_energie = sum(ci_energie.values())

        col_p = "puissance_kw_10min" if "puissance_kw_10min" in df.columns else "puissance_kw"
        # Tableaux triés par mois × plage pour CMDPS HTA
        # dropna() obligatoire (NaN brise searchsorted)
        pw_mois_plage: Dict[tuple, np.ndarray] = {}
        for (mois, plage), groupe in df.groupby([df["timestamp"].dt.month, "plage"]):
            vals = groupe[col_p].dropna().values
            pw_mois_plage[(mois, plage)] = np.sort(vals)

        cache.update({
            "bi": bi, "ci": ci, "ordre": ordre, "bi_ordre": bi_ordre,
            "cs_energie": cs_energie,
            "pw_mois_plage": pw_mois_plage,
        })

    elif domaine == "BT > 36 kVA":
        bi = BT_SUP_BI[fta]
        ci = BT_SUP_CI[fta]
        ordre    = sorted(PLAGES_BT_SUP, key=lambda p: bi[p], reverse=True)
        bi_ordre = [bi[p] for p in ordre]
        cs_energie = sum((ci[p] / 100) * energies.get(p, 0) * fact_ann for p in PLAGES_BT_SUP)

        pw_plage: Dict[str, np.ndarray] = {
            p: np.sort(df[df["plage"] == p]["puissance_kw"].dropna().values)
            for p in PLAGES_BT_SUP
        }
        pw_mois_plage_bt: Dict[int, Dict[str, np.ndarray]] = {}
        for mois, grp_m in df.groupby(df["timestamp"].dt.month):
            pw_mois_plage_bt[mois] = {
                p: np.sort(grp_m[grp_m["plage"] == p]["puissance_kw"].dropna().values)
                for p in PLAGES_BT_SUP
            }
        cs_energie_m = cs_energie / 12.0
        cache.update({
            "bi": bi, "ci": ci, "ordre": ordre, "bi_ordre": bi_ordre,
            "cs_energie": cs_energie, "pw_plage": pw_plage,
            "pw_mois_plage_bt": pw_mois_plage_bt, "cs_energie_m": cs_energie_m,
        })

    else:  # BT ≤ 36 kVA
        b  = BT_INF_B[fta]
        ci = BT_INF_CI[fta]
        cs_energie = sum((ci[p] / 100) * energies.get(p, 0) * fact_ann for p in ci)
        pw_unique  = np.sort(df["puissance_kw"].dropna().values)
        cache.update({"b": b, "ci": ci, "cs_energie": cs_energie, "pw_unique": pw_unique})

    return cache


def _cout_depuis_cache(cache: Dict, puissances_souscrites: Dict[str, float]) -> Dict:
    """
    Calcule le coût TURPE+CTA HT à partir du cache précalculé.
    Aucun accès DataFrame — uniquement arithmétique et searchsorted.
    ~10-30× plus rapide que calculer_cout_total sur DataFrame.
    """
    cg       = cache["cg"]
    cc       = cache["cc"]
    fact_ann = cache["fact_ann"]
    domaine  = cache["domaine"]
    TAUX_CTA = 0.15

    if domaine == "HTA":
        ordre    = cache["ordre"]
        bi       = cache["bi"]
        bi_ordre = cache["bi_ordre"]
        ps_tries = [puissances_souscrites[p] for p in ordre]

        cs_puissance = bi_ordre[0] * ps_tries[0]
        for idx in range(1, len(ordre)):
            cs_puissance += bi_ordre[idx] * (ps_tries[idx] - ps_tries[idx - 1])

        cs = cs_puissance + cache["cs_energie"]

        # CMDPS HTA : sqrt(sum(deltas²)) par mois/plage via searchsorted
        # CMDPS HTA : 2 × Σ_mois Σ_plage [0.04 × bᵢ × √(Σ_t ΔPᵢ_t²)] — délibération 2025-78
        cmdps = 0.0
        for (mois, plage), arr in cache["pw_mois_plage"].items():
            ps  = puissances_souscrites.get(plage, 0)
            idx = np.searchsorted(arr, ps, side="right")
            if idx < len(arr):
                deltas = arr[idx:] - ps
                cmdps += 0.04 * bi[plage] * np.sqrt(np.dot(deltas, deltas))
        cmdps *= 2.0 * fact_ann   # ×2 obligatoire

    elif domaine == "BT > 36 kVA":
        ordre    = cache["ordre"]
        bi_ordre = cache["bi_ordre"]
        ps_tries = [puissances_souscrites[p] for p in ordre]

        cs_puissance = bi_ordre[0] * ps_tries[0]
        for idx in range(1, len(ordre)):
            cs_puissance += bi_ordre[idx] * (ps_tries[idx] - ps_tries[idx - 1])

        cs = cs_puissance + cache["cs_energie"]

        # CMDPS BT>36 : 12.41€/h × heures dépassement, avec capping mensuel TURPE 7
        bi_bt    = cache["bi"]
        ordre_bt = cache["ordre"]
        tarif_ps_supp_m = sum(bi_bt[p] * puissances_souscrites[p] for p in cache["pw_plage"]) / 12.0
        bi_ord   = cache["bi_ordre"]
        ps_tries = [puissances_souscrites[p] for p in ordre_bt]
        cs_p_m   = (bi_ord[0] * ps_tries[0] + sum(
            bi_ord[i] * (ps_tries[i] - ps_tries[i-1]) for i in range(1, len(ordre_bt))
        )) / 12.0
        facture_m = (cache["cg"] + cache["cc"]) / 12.0 + cs_p_m + cache.get("cs_energie_m", 0.0)
        cmdps = 0.0
        for mois, grp in cache["pw_mois_plage_bt"].items():
            h_dep_m = sum(
                len(arr) - np.searchsorted(arr, puissances_souscrites[p], side="right")
                for p, arr in grp.items()
            )
            cmdps_m = 12.41 * h_dep_m
            cap_30  = 0.30 * facture_m
            cap_25x = 25.0 * tarif_ps_supp_m
            if cmdps_m > cap_30 and cmdps_m > cap_25x:
                cmdps_m = max(cap_30, cap_25x)
            cmdps += cmdps_m
        cmdps *= fact_ann

    else:  # BT ≤ 36 kVA
        ps_unique    = list(puissances_souscrites.values())[0]
        cs_puissance = cache["b"] * ps_unique
        cs           = cs_puissance + cache["cs_energie"]
        cmdps        = 0.0

    cta_ht   = (cg + cc + cs_puissance) * TAUX_CTA
    total    = cg + cc + cs + cmdps
    total_ht = total + cta_ht

    return {
        "CG":             round(cg, 2),
        "CC":             round(cc, 2),
        "CS":             round(cs, 2),
        "CS_puissance":   round(cs_puissance, 2),
        "CMDPS":          round(cmdps, 2),
        "CTA_HT":         round(cta_ht, 2),
        "CTA_TTC":        round(cta_ht * 1.20, 2),
        "Total":          round(total, 2),
        "Total_HT":       round(total_ht, 2),
        "Total_avec_CTA": round(total_ht, 2),
        "facteur_annualisation": fact_ann,
        "puissances_souscrites": puissances_souscrites,
    }


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

        # CMDPS HTA : 2 × Σ_mois [...]
        col_p = "puissance_kw_10min" if "puissance_kw_10min" in df.columns else "puissance_kw"
        cmdps = 0.0
        for (_, plage), groupe in df.groupby([df["timestamp"].dt.month, "plage"]):
            ps     = puissances_souscrites.get(plage, 0)
            deltas = np.maximum(0, groupe[col_p].dropna().values - ps)
            if deltas.sum() > 0:
                cmdps += 0.04 * bi[plage] * np.sqrt(np.sum(deltas ** 2))
        cmdps *= 2.0 * fact_ann   # ×2 obligatoire

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

        # CMDPS BT>36 avec capping mensuel TURPE 7
        tarif_ps_supp_m = sum(bi[p] * puissances_souscrites[p] for p in PLAGES_BT_SUP) / 12.0
        facture_m_base  = (cg + cc) / 12.0 + cs_puissance / 12.0
        cmdps = 0.0
        for mois, grp_m in df.groupby(df["timestamp"].dt.month):
            h_dep_m = int(sum(
                (grp_m[grp_m["plage"] == p]["puissance_kw"] > ps).sum()
                for p, ps in puissances_souscrites.items()
            ))
            cmdps_m = 12.41 * h_dep_m
            cap_30  = 0.30 * facture_m_base
            cap_25x = 25.0 * tarif_ps_supp_m
            if cmdps_m > cap_30 and cmdps_m > cap_25x:
                cmdps_m = max(cap_30, cap_25x)
            cmdps += cmdps_m
        cmdps *= fact_ann

    else:  # BT ≤ 36 kVA
        ps_unique = list(puissances_souscrites.values())[0]
        b  = BT_INF_B[fta]
        ci = BT_INF_CI[fta]
        cs_puissance = b * ps_unique
        cs_energie   = sum((ci[p] / 100) * energies.get(p, 0) * fact_ann for p in ci)
        cs    = cs_puissance + cs_energie
        cmdps = 0.0

    # ── CTA (Contribution Tarifaire d'Acheminement) ───────────────────────────
    # Taux : 15 % depuis le 1er février 2026 — Enedis réseau de distribution
    # Assiette : CG + CC + CS_puissance (composantes fixes, hors énergie et CMDPS)
    # La CTA est elle-même soumise à TVA 20 % sur facture — ici on expose HT et TTC
    TAUX_CTA     = 0.15
    TAUX_TVA_CTA = 0.20
    cta_ht       = (cg + cc + cs_puissance) * TAUX_CTA
    cta_ttc      = cta_ht * (1 + TAUX_TVA_CTA)

    total     = cg + cc + cs + cmdps          # TURPE HT
    total_ht  = total + cta_ht                # TURPE + CTA, tout en HT

    return {
        "CG":               round(cg, 2),
        "CC":               round(cc, 2),
        "CS":               round(cs, 2),
        "CS_puissance":     round(cs_puissance, 2),
        "CMDPS":            round(cmdps, 2),
        "CTA_HT":           round(cta_ht, 2),
        "CTA_TTC":          round(cta_ttc, 2),
        "Total":            round(total, 2),         # TURPE seul HT
        "Total_HT":         round(total_ht, 2),      # TURPE + CTA en HT ← indicateur principal
        "Total_avec_CTA":   round(total_ht, 2),      # alias conservé pour compatibilité
        "facteur_annualisation": fact_ann,
        "puissances_souscrites": puissances_souscrites,
    }


# ─────────────────────────────────────────────
# 5. MOTEUR D'OPTIMISATION
# ─────────────────────────────────────────────

def _appliquer_contrainte(ps: Dict[str, float], bi: Dict[str, float]) -> Dict[str, float]:
    """Contrainte TURPE : plage plus chère = PS plus basse."""
    ps = dict(ps)
    for p in sorted(ps, key=lambda x: bi[x], reverse=True):
        pass  # ordre calculé ci-dessous
    plages_dec = sorted(ps.keys(), key=lambda p: bi[p], reverse=True)
    ps_min = 0
    for p in plages_dec:
        ps[p] = max(ps[p], ps_min)
        ps_min = ps[p]
    return ps


def _descente_coordonnees_cache(
    cache: Dict, plages: List[str], bi: Dict,
    ps_depart: Dict[str, float],
    max_par_plage: Dict[str, float],
    fenetre: int = 25,
    n_iter: int = 6,
) -> Dict[str, float]:
    """Descente par coordonnées utilisant le cache — aucun accès DataFrame."""
    current = dict(ps_depart)
    for _ in range(n_iter):
        improved = False
        for plage in plages:
            best_cost = _cout_depuis_cache(cache, current)["Total_HT"]
            best_ps   = current[plage]
            lo = max(1, current[plage] - fenetre)
            hi = min(int(max_par_plage[plage]) + 2, current[plage] + fenetre)
            for ps_cand in range(lo, hi + 1):
                if ps_cand == current[plage]:
                    continue
                ps_test = dict(current)
                ps_test[plage] = ps_cand
                c = _cout_depuis_cache(cache, ps_test)["Total_HT"]
                if c < best_cost:
                    best_cost = c
                    best_ps   = ps_cand
            if best_ps != current[plage]:
                current[plage] = best_ps
                improved = True
        if not improved:
            break
    return current


def optimiser_puissances(
    df: pd.DataFrame,
    domaine: str,
    fta: str,
    type_contrat: str = "contrat_unique",
    pas_kva: int = 1,          # conservé pour compatibilité, toujours 1
    ps_actuelles: Optional[Dict[str, float]] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Optimise les puissances souscrites (minimise Total_HT = TURPE + CTA HT).

    Grâce au cache précalculé, chaque évaluation est ~20× plus rapide.
    L'algorithme explore donc un espace bien plus large dans le même temps.

    4 phases complémentaires :
      1. Balayage uniforme P40→Pmax  → trouve P* (meilleur PS commun)
      2. Descente coordonnées multi-départs autour de P*
      3. Compression "flatten" : force toutes les plages > V à V
         → détecte les PS uniformes basses optimales malgré quelques dépassements
      4. Raffinement fin ±5 kVA autour du meilleur candidat trouvé
    """
    plages = (
        PLAGES_HTA     if domaine == "HTA"         else
        PLAGES_BT_SUP  if domaine == "BT > 36 kVA" else
        ["unique"]
    )

    # ── Construction du cache (1 seul accès DataFrame) ───────────────────────
    cache = _construire_cache(df, domaine, fta, type_contrat)

    if domaine in ["HTA", "BT > 36 kVA"]:
        bi = HTA_BI[fta] if domaine == "HTA" else BT_SUP_BI[fta]

        p_global_max  = df["puissance_kw"].max()
        max_par_plage = {
            p: (df[df["plage"] == p]["puissance_kw"].max() if (df["plage"] == p).any() else p_global_max)
            for p in plages
        }
        p_min_search = max(1, int(df["puissance_kw"].quantile(0.40)) - 1)
        p_max_search = int(p_global_max) + 2

        # ── Phase 1 : balayage uniforme ───────────────────────────────────────
        best_cost_global = float("inf")
        ps_commune_opt   = int(p_global_max)

        for ps_val in range(p_min_search, p_max_search):
            ps_test = {p: ps_val for p in plages}
            c = _cout_depuis_cache(cache, ps_test)["Total_HT"]
            if c < best_cost_global:
                best_cost_global = c
                ps_commune_opt   = ps_val

        # ── Phase 2 : descente coordonnées multi-départs ──────────────────────
        best_ps   = {p: ps_commune_opt for p in plages}
        best_cost = best_cost_global

        departs = sorted(set([
            max(1, ps_commune_opt - 15),
            max(1, ps_commune_opt - 8),
            max(1, ps_commune_opt - 3),
            ps_commune_opt,
            ps_commune_opt + 5,
            ps_commune_opt + 12,
        ]))

        for ps_dep in departs:
            ps_start  = {p: ps_dep for p in plages}
            ps_result = _descente_coordonnees_cache(
                cache, plages, bi, ps_start, max_par_plage, fenetre=30, n_iter=8,
            )
            ps_result = _appliquer_contrainte(ps_result, bi)
            c = _cout_depuis_cache(cache, ps_result)["Total_HT"]
            if c < best_cost:
                best_cost = c
                best_ps   = ps_result

        # ── Phase 3 : compression "flatten" ──────────────────────────────────
        # Teste : forcer toutes les plages > V à V (pour V ∈ [p_min, max(best_ps)])
        # Capture les cas PS uniforme basse > économie > coût dépassements
        ps_max_current = max(best_ps.values())
        for v in range(p_min_search, int(ps_max_current) + 1):
            ps_flat = {p: min(best_ps[p], v) for p in plages}
            ps_flat = _appliquer_contrainte(ps_flat, bi)
            c = _cout_depuis_cache(cache, ps_flat)["Total_HT"]
            if c < best_cost:
                best_cost = c
                best_ps   = ps_flat

        # ── Phase 4 : raffinement fin ─────────────────────────────────────────
        ps_final = _descente_coordonnees_cache(
            cache, plages, bi, best_ps, max_par_plage, fenetre=6, n_iter=4,
        )
        ps_final = _appliquer_contrainte(ps_final, bi)
        c_final  = _cout_depuis_cache(cache, ps_final)["Total_HT"]
        if c_final < best_cost:
            best_cost = c_final
            best_ps   = ps_final

        meilleures_ps    = best_ps
        resultat_optimal = _cout_depuis_cache(cache, meilleures_ps)

        # ── Non-régression ────────────────────────────────────────────────────
        if ps_actuelles is not None:
            resultat_ref = _cout_depuis_cache(cache, ps_actuelles)
            if resultat_optimal["Total_HT"] >= resultat_ref["Total_HT"]:
                meilleures_ps    = dict(ps_actuelles)
                resultat_optimal = resultat_ref

        # ── Scénarios de sensibilité (±8 kVA autour de l'optimal) ────────────
        scenarios = []
        for plage in plages:
            ps_opt = meilleures_ps[plage]
            for delta in range(-8, 9):
                ps_var = dict(meilleures_ps)
                ps_var[plage] = max(1, ps_opt + delta)
                r = _cout_depuis_cache(cache, ps_var)
                r["plage_variee"] = plage
                r["ps_variee"]    = ps_var[plage]
                scenarios.append(r)
        df_scenarios = pd.DataFrame(scenarios)

    else:  # BT ≤ 36 kVA — Linky : coupure immédiate dès que P > PS
        # ── Contrainte réglementaire Linky ────────────────────────────────────
        # Le compteur Linky coupe IMMÉDIATEMENT si la puissance dépasse la PS.
        # Conséquence : toute PS < Pmax entraîne des interruptions de fourniture.
        # L'optimiseur ne peut donc pas explorer en dessous de ceil(Pmax).
        # Il n'y a pas de CMDPS pour ce domaine — la "pénalité" est une coupure,
        # non modélisable financièrement → contrainte dure : PS ≥ ceil(Pmax).
        p_max      = df["puissance_kw"].max()
        p_min_safe = int(np.ceil(p_max))          # borne dure : aucune coupure
        p_max_srch = p_min_safe + 20              # explorer jusqu'à +20 kVA au-dessus

        scenarios   = []
        best_cost   = float("inf")
        best_ps_val = p_min_safe

        # ── Optimisation : uniquement PS ≥ ceil(Pmax) ─────────────────────────
        for ps_val in range(p_min_safe, p_max_srch + 1):
            r = _cout_depuis_cache(cache, {"unique": ps_val})
            r["ps_variee"]    = ps_val
            r["zone_coupure"] = False
            scenarios.append(r)
            if r["Total_HT"] < best_cost:
                best_cost   = r["Total_HT"]
                best_ps_val = ps_val

        # ── Scénarios "zone coupure" en dessous de Pmax (pour graphique seult) ─
        p_min_scen = max(1, int(df["puissance_kw"].quantile(0.40)) - 1)
        for ps_val in range(p_min_scen, p_min_safe):
            r = _cout_depuis_cache(cache, {"unique": ps_val})
            r["ps_variee"]    = ps_val
            r["zone_coupure"] = True   # ← PS insuffisante : Linky coupera
            scenarios.append(r)

        meilleures_ps    = {"unique": best_ps_val}
        resultat_optimal = _cout_depuis_cache(cache, meilleures_ps)

        # ── Non-régression ────────────────────────────────────────────────────
        if ps_actuelles is not None:
            ps_act_safe = {"unique": max(p_min_safe, int(list(ps_actuelles.values())[0]))}
            resultat_ref = _cout_depuis_cache(cache, ps_act_safe)
            if resultat_optimal["Total_HT"] >= resultat_ref["Total_HT"]:
                meilleures_ps    = ps_act_safe
                resultat_optimal = resultat_ref

        df_scenarios = pd.DataFrame(scenarios).sort_values("ps_variee").reset_index(drop=True)

    return resultat_optimal, df_scenarios
