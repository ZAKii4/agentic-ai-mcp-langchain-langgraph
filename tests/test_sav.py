"""Tests déterministes du SAV (sans LLM).  Lancement : uv run pytest -q"""
import sqlite3
from datetime import date

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from d_sav.garantie import lire_garantie, rediger_verdict
from d_sav.generer_donnees import REGISTRE
from d_sav.orchestrator import conclure_documentaire, conclure_garantie, orchestrateur
from d_sav.rag.agent_rag import chemin_photo

JOUR = date(2026, 11, 26)


@pytest.mark.parametrize("numero, statut", [
    ("PX200-0042", "active"), (" px200-0042 ", "active"), ("PX200-0007", "expiree"),
    ("PX200-0099", "annulee"), ("PX200-9999", "inconnu"), ("' OR 1=1 --", "numero_invalide")])
def test_statuts(numero, statut):
    resultat = lire_garantie(numero, JOUR)
    assert resultat["statut"] == statut
    assert resultat["numero_serie"] in rediger_verdict(resultat)


def test_dernier_jour_couvert():
    assert lire_garantie("PX200-0042", date(2027, 3, 13))["statut"] == "active"
    assert lire_garantie("PX200-0042", date(2027, 3, 14))["statut"] == "expiree"


def test_registre_en_lecture_seule():
    base = sqlite3.connect(f"{REGISTRE.as_uri()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        base.execute("DELETE FROM garanties")
    base.close()


@pytest.mark.parametrize("reference", ["../../.env", "/etc/passwd", "img_inexistante", "IMG_X"])
def test_reference_photo_refusee(reference):
    with pytest.raises(ValueError):
        chemin_photo(reference)


def test_orchestrateur_termine_quand_la_reponse_existe():
    commande = orchestrateur({"messages": [HumanMessage("q"), ("ai", "r")], "reponse_finale": "r"})
    assert commande.goto == "__end__"


def test_numero_invente_par_le_llm_ignore():
    observation = ToolMessage('{"statut": "active", "numero_serie": "PX200-0042"}',
                              name="consulter_garantie", tool_call_id="1")
    sortie = conclure_garantie({"question": "Ma garantie est-elle valable ?",
                                "messages": [HumanMessage("Ma garantie ?"), observation]})
    assert "indiquez le numéro de série" in sortie["reponse_finale"]


def test_panne_mcp():
    erreur = ToolMessage("Serveur MCP indisponible", name="rag_multimodal", status="error", tool_call_id="1")
    sortie = conclure_documentaire({"messages": [HumanMessage("q"), erreur]})
    assert "indisponible" in sortie["reponse_finale"]
