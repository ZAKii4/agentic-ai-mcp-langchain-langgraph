"""Index texte (ChromaDB + mxbai-embed-large via Ollama) d'un PDF, partagé par les chatbots RAG."""
import hashlib
from functools import lru_cache
from pathlib import Path

import chromadb
import pymupdf
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

RACINE = Path(__file__).resolve().parents[1]
PDF_PAR_DEFAUT = RACINE / "data" / "attention_is_all_you_need.pdf"
DOSSIER_CHROMA = RACINE / "chroma_db"
PREFIXE_REQUETE = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def embeddings():
    return OllamaEmbeddings(model="mxbai-embed-large")


@lru_cache(maxsize=4)
def indexer(pdf: Path = PDF_PAR_DEFAUT):
    """Indexe le PDF une seule fois ; le nom de la collection dépend de son contenu (SHA-256)."""
    empreinte = hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]
    collection = chromadb.PersistentClient(path=str(DOSSIER_CHROMA)).get_or_create_collection(
        f"texte_{empreinte}", metadata={"hnsw:space": "cosine"})
    decoupeur = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    ids, documents, metadatas = [], [], []
    with pymupdf.open(pdf) as doc:
        for page, contenu in enumerate(doc, start=1):
            for i, fragment in enumerate(decoupeur.split_text(contenu.get_text(sort=True))):
                ids.append(f"{page}:{i}")
                documents.append(fragment)
                metadatas.append({"page": page, "pdf": pdf.name})
    if collection.count() != len(ids):
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas,
                          embeddings=embeddings().embed_documents(documents))
    return collection


def rechercher(requete: str, k: int = 4, pdf: Path = PDF_PAR_DEFAUT) -> list[dict]:
    r = indexer(pdf).query(query_embeddings=[embeddings().embed_query(PREFIXE_REQUETE + requete)],
                           n_results=k)
    return [{"contenu": d, "page": m["page"], "score": 1 - dist}
            for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]


def formater(resultats: list[dict]) -> str:
    return "\n\n".join(f"[page {r['page']}]\n{r['contenu']}" for r in resultats)
