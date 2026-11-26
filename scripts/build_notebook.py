"""Convertit un source au format « percent » (# %% / # %% [markdown]) en notebook .ipynb."""
import sys
from pathlib import Path

import nbformat as nbf

source, cible = Path(sys.argv[1]), Path(sys.argv[2])
cellules, genre, lignes = [], None, []


def vider():
    texte = "\n".join(lignes).strip("\n")
    if genre == "markdown":
        texte = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in texte.split("\n"))
        cellules.append(nbf.v4.new_markdown_cell(texte))
    elif genre == "code" and texte:
        cellules.append(nbf.v4.new_code_cell(texte))


for ligne in source.read_text(encoding="utf-8").split("\n"):
    if ligne.startswith("# %%"):
        vider()
        genre, lignes = ("markdown" if "[markdown]" in ligne else "code"), []
    else:
        lignes.append(ligne)
vider()
nb = nbf.v4.new_notebook(cells=cellules)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, cible)
print(cible, len(cellules), "cellules")
