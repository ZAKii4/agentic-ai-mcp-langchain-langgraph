# Activité Pratique N°5 : Agentic AI, LangChain, LangGraph et MCP

## Installation

```bash
uv sync
cp .env.example .env          # seulement si .env n'existe pas encore
ollama pull qwen3:8b && ollama pull qwen3:4b && ollama pull mxbai-embed-large
```

Tout fonctionne en local avec Ollama. OpenAI (`LLM_PROVIDER=openai`) et Tavily (`TAVILY_API_KEY`) sont
optionnels ; l'index de recherche utilise Ollama dans tous les cas.
Les poids CLIP (partie D) sont téléchargés au premier lancement.

## A. LangChain

- Notebook : `a_langchain/01_agent_langchain.ipynb` (agent, mémoire, tools, DuckDuckGo, PythonREPLTool, middlewares)
- Chatbot RAG agentique : `uv run python -m a_langchain.rag_chatbot`

Les notebooks sont générés à partir des fichiers `.src.py` :
`uv run python scripts/build_notebook.py <source.src.py> <notebook.ipynb>`

## B. LangGraph

- Bases : `b_langgraph/01_bases_langgraph.ipynb`
- LangGraph Studio : `uv run langgraph dev` (graphes `agentic_rag` et `sav`)
- Interface web : `uv run streamlit run b_langgraph/app_streamlit.py`

## C. MCP

Le serveur expose `get_employee_info` et `get_weather` (données de démonstration).

```bash
uv run python -m c_mcp.server                 # terminal 1 : http://127.0.0.1:8000/mcp
uv run python -m c_mcp.agent_client           # terminal 2 : agent LangChain
npx -y @modelcontextprotocol/inspector        # interface : Streamable HTTP + URL ci-dessus
```

**n8n** : lancer `npx n8n` (Node ≥ 24), puis :

1. Importer `c_mcp/n8n_agent_mcp.json` dans un workflow.
2. Dans le nœud *Ollama Chat Model*, créer l'identifiant avec l'URL `http://127.0.0.1:11434`.
3. Ouvrir le chat et demander : « Quel est le salaire de Fatima, et quel temps fait-il à Rabat ? »

Avec n8n dans Docker, `127.0.0.1` désigne le conteneur : il faut `host.docker.internal` dans les deux URL
et `MCP_HOST=0.0.0.0` pour le serveur. Ce serveur n'a pas d'authentification et devient alors visible sur
le réseau. Cette variante Docker n'a pas été testée.

## D. Synthèse : SAV multi-agents

```
START → orchestrateur ─documentation→ agent_documentaire (llm ⇄ tools) ──MCP──> rag_multimodal
             │      └─garantie─────→ agent_garantie     (llm ⇄ tools) ──────> consulter_garantie
             │ ←───────────── retour des agents ─────────────┘
             └─ réponse finale disponible → END
```

```bash
uv run python -m d_sav.mcp_server             # terminal 1 : http://127.0.0.1:8001/mcp
# terminal 2
uv run python -m d_sav.orchestrator "Mon imprimante PX-200 affiche un voyant clignotant. Que signifie le pictogramme sur la photo jointe ?" --photo img_px200_01
uv run python -m d_sav.orchestrator "Est-ce que mon imprimante est encore sous garantie ? Son numéro de série est PX200-0042."
uv run pytest -q
```

Dans Studio, la photo se renseigne dans le champ `photo_ref`.

Garde-fous assurés par le code :

- l'agent documentaire appelle toujours l'outil, avec la question et la photo d'origine ;
- les sources viennent des éléments retrouvés ; sans citation, elles sont notées « consultées » ;
- une référence de photo n'est jamais un chemin de fichier ;
- le registre est ouvert en lecture seule et le verdict de garantie est rédigé par le code ;
- un numéro de série absent de la question du client est ignoré ;
- si le serveur MCP est arrêté, le client reçoit un message d'indisponibilité.

## Données fictives

Le manuel PX-200, les 4 photos et le registre de garantie sont **inventés** pour la démonstration
(`uv run python -m d_sav.generer_donnees` les recrée).

## Limites

- Les photos sont simulées à partir des mêmes dessins que les figures du manuel. La reconnaissance de
  vraies photos n'est **pas** démontrée, et les seuils CLIP (`agent_rag.py`) sont à recalibrer.
- `mxbai-embed-large` est surtout anglophone. Sur le manuel en français, la bonne page était classée
  dernière pour une question sur le réseau ; une recherche hybride (sens + mots-clés) corrige ce cas.
  Elle a été vérifiée sur 12 questions d'un manuel de 4 pages, ce qui est trop peu pour généraliser.
- Les petits modèles locaux peuvent se tromper, y compris de numéro de page (observé une fois sur le
  RAG de l'article : p. 4 au lieu de p. 5). Les pages citées sont à vérifier.
- `PythonREPLTool` exécute du code arbitraire : le dossier `sandbox/` n'est pas une isolation.
- La météo et les employés sont des données simulées.
