"""Génère les données FICTIVES du SAV : manuel PX-200 (PDF avec figures), photos et registre de garantie.

Lancement : uv run python -m d_sav.generer_donnees
"""
import io
import random
import sqlite3
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFilter

DATA = Path(__file__).resolve().parent / "data"
MANUEL = DATA / "manuel_px200.pdf"
PHOTOS = DATA / "photos"
REGISTRE = DATA / "registre_garantie.db"


def pictogramme(nom: str, trait="black", fond="white") -> Image.Image:
    im = Image.new("RGB", (400, 400), fond)
    d = ImageDraw.Draw(im)

    def ligne(*points, largeur=14):
        d.line(points, fill=trait, width=largeur, joint="curve")

    if nom == "bourrage":  # feuille froissée coincée entre deux rouleaux
        ligne((130, 150), (130, 40), (270, 40), (270, 150))
        ligne((130, 150), (175, 185), (140, 220), (200, 245), (245, 205), (225, 175), (270, 150))
        d.ellipse((60, 260, 180, 380), outline=trait, width=14)
        d.ellipse((220, 260, 340, 380), outline=trait, width=14)
    elif nom == "encre":  # goutte presque vide
        ligne((200, 40), (100, 220))
        ligne((200, 40), (300, 220))
        d.arc((95, 150, 305, 360), start=-12, end=192, fill=trait, width=14)
        d.chord((120, 175, 280, 335), start=40, end=140, fill=trait)
    elif nom == "wifi":  # ondes barrées
        for r in (60, 120, 180):
            d.arc((200 - r, 300 - r, 200 + r, 300 + r), start=225, end=315, fill=trait, width=18)
        d.ellipse((180, 280, 220, 320), fill=trait)
        ligne((80, 80), (330, 340), largeur=18)
    elif nom == "capot":  # imprimante au capot ouvert
        d.rectangle((60, 210, 340, 340), outline=trait, width=14)
        ligne((60, 210), (110, 70), (370, 120), (340, 210))
        d.rectangle((110, 260, 290, 290), fill=trait)
    else:
        raise ValueError(nom)
    return im


FICHES = [
    ("bourrage", "Figure 1 : pictogramme bourrage papier",
     "Code E01 - Bourrage papier",
     "Le pictogramme représente une feuille froissée entre deux rouleaux. Le voyant Alerte clignote en orange. "
     "Une feuille est coincée dans le circuit papier.\n\nDépannage :\n1. Éteignez l'imprimante et débranchez-la.\n"
     "2. Ouvrez le capot arrière.\n3. Retirez la feuille lentement, à deux mains, dans le sens de circulation du papier. "
     "N'utilisez aucun outil.\n4. Vérifiez qu'aucun morceau déchiré ne reste.\n5. Refermez le capot, rebranchez et "
     "appuyez sur OK. Si le voyant clignote toujours, contactez le service après-vente."),
    ("encre", "Figure 2 : pictogramme niveau d'encre faible",
     "Code E02 - Niveau d'encre faible",
     "Le pictogramme représente une goutte presque vide. Le voyant Encre clignote lentement. Une cartouche est "
     "presque épuisée.\n\nDépannage :\n1. Laissez l'imprimante allumée et ouvrez le capot avant.\n"
     "2. Attendez l'arrêt du chariot.\n3. Remplacez la cartouche signalée par une cartouche compatible PX-200.\n"
     "4. Refermez le capot et lancez l'alignement depuis le menu Maintenance."),
    ("wifi", "Figure 3 : pictogramme connexion Wi-Fi perdue",
     "Code E03 - Connexion Wi-Fi perdue",
     "Le pictogramme représente des ondes Wi-Fi barrées. Le voyant Réseau clignote en bleu. L'imprimante n'est plus "
     "connectée au réseau.\n\nDépannage :\n1. Vérifiez que la box est allumée.\n2. Rapprochez l'imprimante de la box.\n"
     "3. Maintenez la touche Wi-Fi pendant 3 secondes, puis appuyez sur le bouton WPS de la box."),
    ("capot", "Figure 4 : pictogramme capot ouvert",
     "Code E04 - Capot ouvert",
     "Le pictogramme représente une imprimante dont le capot est levé. Le voyant Alerte reste allumé sans clignoter. "
     "L'impression est suspendue.\n\nDépannage :\n1. Fermez le capot avant jusqu'au déclic.\n"
     "2. Vérifiez qu'aucun objet n'empêche la fermeture."),
]


def png(image: Image.Image) -> bytes:
    tampon = io.BytesIO()
    image.save(tampon, "PNG")
    return tampon.getvalue()


def creer_manuel():
    doc = pymupdf.open()
    for nom, legende, titre, texte in FICHES:
        page = doc.new_page()
        page.insert_text((60, 50), "Imprimante PX-200 - Manuel de dépannage (document fictif)", fontsize=9)
        page.insert_text((60, 90), titre, fontsize=17, fontname="hebo")
        page.insert_image(pymupdf.Rect(197, 115, 397, 315), stream=png(pictogramme(nom)))
        page.insert_text((200, 335), legende, fontsize=10, fontname="heit")
        reste = page.insert_textbox(pymupdf.Rect(60, 360, 535, 780), texte, fontsize=11.5)
        assert reste >= 0, f"texte trop long : {titre}"
    doc.save(MANUEL, no_new_id=True)


def creer_photo(nom: str, destination: Path, angle: float, graine: int):
    """Simule la photo d'un écran : boîtier sombre, écran ambré, inclinaison, flou et bruit."""
    alea = random.Random(graine)
    photo = Image.new("RGB", (900, 700), (52, 54, 58))
    d = ImageDraw.Draw(photo)
    d.rounded_rectangle((130, 90, 770, 590), radius=30, fill=(20, 20, 22))
    d.rectangle((190, 135, 710, 545), fill=(236, 214, 150))
    icone = pictogramme(nom, trait=(45, 32, 10), fond=(236, 214, 150)).resize((340, 340))
    photo.paste(icone, (280, 170))
    d.ellipse((800, 330, 832, 362), fill=(255, 140, 0))
    d.text((330, 620), "PX-200", fill=(215, 215, 215), font_size=46)
    photo = photo.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(52, 54, 58))
    pixels = photo.load()
    for _ in range(25000):
        x, y = alea.randrange(900), alea.randrange(700)
        ecart = alea.randint(-30, 30)
        pixels[x, y] = tuple(max(0, min(255, c + ecart)) for c in pixels[x, y])
    photo.filter(ImageFilter.GaussianBlur(1.1)).save(destination)


# numéro de série, produit, date d'achat, fin de garantie, annulée (0/1), motif
GARANTIES = [
    ("PX200-0042", "Imprimante PX-200", "2025-03-14", "2027-03-13", 0, None),
    ("PX200-0007", "Imprimante PX-200", "2023-05-02", "2025-05-01", 0, None),
    ("PX200-0099", "Imprimante PX-200", "2026-01-10", "2028-01-09", 1, "dommage causé par un liquide"),
]


def creer_registre():
    REGISTRE.unlink(missing_ok=True)
    with sqlite3.connect(REGISTRE) as base:
        base.execute("CREATE TABLE garanties (numero_serie TEXT PRIMARY KEY, produit TEXT, date_achat TEXT,"
                     " fin_garantie TEXT, annulee INTEGER, motif TEXT)")
        base.executemany("INSERT INTO garanties VALUES (?,?,?,?,?,?)", GARANTIES)
    base.close()


if __name__ == "__main__":
    PHOTOS.mkdir(parents=True, exist_ok=True)
    creer_manuel()
    creer_photo("bourrage", PHOTOS / "img_px200_01.png", angle=5, graine=1)
    creer_photo("encre", PHOTOS / "img_px200_02.png", angle=-6, graine=2)
    creer_photo("wifi", PHOTOS / "img_px200_03.png", angle=4, graine=3)
    creer_photo("capot", PHOTOS / "img_px200_04.png", angle=-3, graine=4)
    creer_registre()
    print("Données générées dans", DATA)
