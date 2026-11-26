"""Outil local consulter_garantie : lit un registre SQLite ouvert en lecture seule."""
import re
import sqlite3
from contextlib import closing
from datetime import date

from langchain_core.tools import tool

from d_sav.generer_donnees import REGISTRE

MOTIF_SERIE = re.compile(r"[A-Z0-9]{2,6}-\d{4}")


def lire_garantie(numero_serie: str, aujourdhui: date | None = None) -> dict:
    numero = numero_serie.strip().upper()
    if not MOTIF_SERIE.fullmatch(numero):
        return {"statut": "numero_invalide", "numero_serie": numero}
    with closing(sqlite3.connect(f"{REGISTRE.as_uri()}?mode=ro", uri=True)) as base:
        ligne = base.execute("SELECT produit, date_achat, fin_garantie, annulee, motif FROM garanties "
                             "WHERE numero_serie = ?", (numero,)).fetchone()
    if ligne is None:
        return {"statut": "inconnu", "numero_serie": numero}
    produit, achat, fin, annulee, motif = ligne
    jours = (date.fromisoformat(fin) - (aujourdhui or date.today())).days
    statut = "annulee" if annulee else "active" if jours >= 0 else "expiree"
    return {"statut": statut, "numero_serie": numero, "produit": produit, "date_achat": achat,
            "fin_garantie": fin, "jours_restants": max(jours, 0) if statut == "active" else 0, "motif": motif}


@tool
def consulter_garantie(numero_serie: str) -> dict:
    """Check the warranty of a device from its serial number (format like PX200-0042). Read-only."""
    return lire_garantie(numero_serie)


def rediger_verdict(r: dict) -> str:
    """Réponse construite par le code : le LLM ne peut pas déformer une date ou un statut."""
    n = r["numero_serie"]
    if r["statut"] == "numero_invalide":
        return f"Le numéro de série « {n} » n'a pas un format valide (exemple : PX200-0042)."
    if r["statut"] == "inconnu":
        return f"Le numéro de série {n} est absent du registre. Vérifiez-le ou contactez le service après-vente."
    jour = lambda iso: date.fromisoformat(iso).strftime("%d/%m/%Y")
    entete = f"{r['produit']} ({n}), achat le {jour(r['date_achat'])} : "
    if r["statut"] == "active":
        return entete + f"sous garantie jusqu'au {jour(r['fin_garantie'])} ({r['jours_restants']} jours restants)."
    if r["statut"] == "expiree":
        return entete + f"la garantie a expiré le {jour(r['fin_garantie'])}."
    return entete + f"la garantie est annulée (motif : {r['motif']})."
