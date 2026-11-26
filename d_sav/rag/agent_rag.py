"""Agent RAG multimodal caché derrière l'outil MCP rag_multimodal.

1. La photo (désignée par une référence) est comparée aux figures du manuel avec CLIP.
2. Un premier jeu de passages est recherché ; l'agent peut relancer d'autres recherches.
3. Les sources sont construites par le code à partir des éléments réellement retrouvés.
"""
import re

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool
from PIL import Image

from common.llm import get_llm
from common.pdf_index import formater
from d_sav.generer_donnees import MANUEL, PHOTOS
from d_sav.rag.index import chercher_figures, chercher_passages

MOTIF_REFERENCE = re.compile(r"img_[a-z0-9_]+")
# Seuils réglés sur les 4 photos simulées : à recalibrer avec de vraies photos.
SCORE_MIN, ECART_MIN = 0.40, 0.03

CONSIGNES = """Tu es l'agent RAG du service après-vente. Réponds en français à la question de dépannage.
- Utilise UNIQUEMENT les passages et les figures fournis ou retournés par search_manual.
- Si les passages ne suffisent pas, appelle search_manual avec une autre requête en français.
- Donne la signification, puis les étapes de dépannage. Cite les pages comme (p. 2).
- Si l'information n'est pas dans le manuel, dis-le clairement. N'invente rien."""


def chemin_photo(reference: str):
    """Une référence est un identifiant, jamais un chemin : pas d'accès arbitraire aux fichiers."""
    if not MOTIF_REFERENCE.fullmatch(reference):
        raise ValueError(f"Référence de photo invalide : {reference!r}")
    chemin = PHOTOS / f"{reference}.png"
    if not chemin.is_file():
        raise ValueError(f"Photo introuvable : {reference}")
    return chemin


def identifier_figure(reference: str) -> dict | None:
    """Figure du manuel correspondant à la photo, ou None si la correspondance est douteuse."""
    with Image.open(chemin_photo(reference)) as photo:
        candidats = chercher_figures(photo, k=2)
    meilleur = candidats[0]
    ecart = meilleur["score"] - candidats[1]["score"] if len(candidats) > 1 else 1.0
    return meilleur if meilleur["score"] >= SCORE_MIN and ecart >= ECART_MIN else None


def repondre(question: str, photo_ref: str = "") -> dict:
    figure = None
    if photo_ref:
        try:
            figure = identifier_figure(photo_ref)
        except ValueError as erreur:
            return {"statut": "erreur", "reponse": str(erreur), "sources": []}
        if figure is None:
            return {"statut": "photo_non_identifiee", "sources": [],
                    "reponse": "Le pictogramme de la photo n'a pas pu être identifié de façon fiable. "
                               "Envoyez une photo plus nette de l'écran ou décrivez le symbole."}

    passages = {}

    def chercher(requete: str, k: int = 2) -> list[dict]:
        resultats = chercher_passages(requete, k=k)
        for r in resultats:
            passages[(r["page"], r["contenu"])] = r
        return resultats

    @tool
    def search_manual(query: str) -> str:
        """Search passages of the PX-200 troubleshooting manual. Query in French."""
        return formater(chercher(query))

    contexte = []
    if figure:
        # La page de la figure identifiée fait partie des preuves : on la recherche avec sa légende.
        trouves = [r for r in chercher(figure["legende"], k=4) if r["page"] == figure["page"]]
        passages = {(r["page"], r["contenu"]): r for r in trouves}
        contexte.append(f"Figure reconnue sur la photo : « {figure['legende']} » (p. {figure['page']})")
        contexte.append(formater(trouves))
    else:
        contexte.append(formater(chercher(question)))

    agent = create_agent(get_llm(), [search_manual], system_prompt=CONSIGNES,
                         middleware=[ToolCallLimitMiddleware(run_limit=2, exit_behavior="continue")])
    etat = agent.invoke({"messages": [{"role": "user", "content":
                        f"Question : {question}\n\n" + "\n\n".join(contexte)}]}, {"recursion_limit": 12})
    reponse = etat["messages"][-1].text

    # Sources = pages citées parmi celles retrouvées ; sans citation, elles sont seulement « consultées ».
    retrouvees = {page for page, _ in passages} | ({figure["page"]} if figure else set())
    citees = {int(n) for n in re.findall(r"p\.\s*(\d+)", reponse)} & retrouvees
    sources = [{"type": "passage" if citees else "consulte", "manuel": MANUEL.name, "page": p}
               for p in sorted(citees or retrouvees)]
    if figure:
        sources.insert(0, {"type": "figure", "manuel": MANUEL.name, "page": figure["page"],
                           "figure": figure["legende"], "similarite": round(figure["score"], 3)})
    return {"statut": "ok", "reponse": reponse, "sources": sources}
