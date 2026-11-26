"""Fabrique de modèles : Ollama (local, par défaut) ou OpenAI, choisi par LLM_PROVIDER."""
import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
_DEFAUTS = {
    "ollama": {"smart": "qwen3:8b", "basic": "qwen3:4b", "vision": "qwen2.5vl:3b"},
    "openai": {"smart": "gpt-4o", "basic": "gpt-4o-mini", "vision": "gpt-4o-mini"},
}
if PROVIDER not in _DEFAUTS:
    raise ValueError(f"LLM_PROVIDER inconnu : {PROVIDER!r} (ollama ou openai)")


def nom_modele(niveau: str = "smart") -> str:
    return os.getenv(f"MODEL_{niveau.upper()}") or _DEFAUTS[PROVIDER][niveau]


def get_llm(niveau: str = "smart", temperature: float = 0, **options):
    """niveau : 'smart', 'basic' ou 'vision'."""
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=nom_modele(niveau), temperature=temperature, **options)

    from langchain_ollama import ChatOllama

    if niveau != "vision":
        # qwen3:8b : on coupe la réflexion. qwen3:4b l'ignore et l'écrit dans la réponse :
        # reasoning=True la place alors à part (additional_kwargs).
        options.setdefault("reasoning", nom_modele(niveau) == "qwen3:4b")
        options.setdefault("num_ctx", 8192)
    return ChatOllama(model=nom_modele(niveau), temperature=temperature, **options)
