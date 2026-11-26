"""Index multimodal du manuel : texte (mxbai) et figures (CLIP), stockés dans ChromaDB."""
import hashlib
import io
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import chromadb
import pymupdf
from PIL import Image

from common import pdf_index
from d_sav.generer_donnees import DATA, MANUEL

DOSSIER_INDEX = DATA / "index"


@lru_cache(maxsize=1)
def _clip():
    import open_clip

    modele, _, pretraitement = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    return modele.eval(), pretraitement


def encoder_image(image: Image.Image) -> list[float]:
    import torch

    modele, pretraitement = _clip()
    cote = max(image.size)  # mise au carré : le recadrage central ne coupe rien
    carre = Image.new("RGB", (cote, cote), "white")
    carre.paste(image.convert("RGB"), ((cote - image.width) // 2, (cote - image.height) // 2))
    with torch.inference_mode():
        v = modele.encode_image(pretraitement(carre).unsqueeze(0))
    return (v / v.norm(dim=-1, keepdim=True))[0].tolist()


@lru_cache(maxsize=1)
def figures():
    """Collection des figures du manuel ; son nom dépend du contenu du PDF."""
    empreinte = hashlib.sha256(MANUEL.read_bytes()).hexdigest()[:16]
    collection = chromadb.PersistentClient(path=str(DOSSIER_INDEX)).get_or_create_collection(
        f"figures_{empreinte}", metadata={"hnsw:space": "cosine"})
    if collection.count():
        return collection
    with pymupdf.open(MANUEL) as doc:
        for numero, page in enumerate(doc, start=1):
            legendes = [" ".join(b[4].split()) for b in page.get_text("blocks") if b[4].startswith("Figure ")]
            for rang, infos in enumerate(page.get_images(full=True)):
                image = Image.open(io.BytesIO(doc.extract_image(infos[0])["image"]))
                legende = legendes[rang] if rang < len(legendes) else f"Figure sans légende, page {numero}"
                collection.upsert(ids=[f"{numero}:{rang}"], documents=[legende],
                                  embeddings=[encoder_image(image)],
                                  metadatas=[{"page": numero, "pdf": MANUEL.name}])
    return collection


def chercher_figures(image: Image.Image, k: int = 3) -> list[dict]:
    r = figures().query(query_embeddings=[encoder_image(image)], n_results=k)
    return [{"legende": d, "page": m["page"], "score": 1 - dist}
            for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]


def racines(texte: str) -> set[str]:
    """Mots d'au moins 4 lettres, sans accents, tronqués à 6 lettres (racinisation grossière)."""
    sans_accents = unicodedata.normalize("NFKD", texte.lower()).encode("ascii", "ignore").decode()
    return {mot[:6] for mot in re.findall(r"[a-z0-9-]{4,}", sans_accents)}


def chercher_passages(requete: str, k: int = 2) -> list[dict]:
    """Recherche hybride : fusion des rangs sémantique (mxbai) et lexical.

    mxbai-embed-large est surtout anglophone : sur le manuel en français, les mots-clés compensent.
    """
    tous = pdf_index.rechercher(requete, k=pdf_index.indexer(Path(MANUEL)).count(), pdf=Path(MANUEL))
    mots = [racines(r["contenu"]) for r in tous]

    def poids(mot):  # un mot présent partout (« imprimante ») ne discrimine pas
        return math.log(len(tous) / max(1, sum(mot in m for m in mots)))

    lexical = [sum(poids(mot) for mot in racines(requete) & m) for m in mots]
    ordre = sorted(range(len(tous)), key=lambda i: -lexical[i])
    rang_lexical = {i: rang for rang, i in enumerate(ordre) if lexical[i] > 0}

    def fusion(i):  # Reciprocal Rank Fusion ; « tous » est déjà trié par score sémantique
        return 1 / (60 + i) + (1 / (60 + rang_lexical[i]) if i in rang_lexical else 0)

    return [tous[i] for i in sorted(range(len(tous)), key=lambda i: -fusion(i))[:k]]
