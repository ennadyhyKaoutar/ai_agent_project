from pdf2image import convert_from_path
import cv2
import numpy as np
from google import genai
from google.genai import types
import io
from PIL import Image
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()  # Charge les variables du fichier .env
# Chemin vers Poppler sur ton PC
poppler_path = r"C:\Release-25.12.0-0\poppler-25.12.0\Library\bin"  # <-- mets ici ton chemin exact


# we convert our pdf into an image
images = convert_from_path("./QA_IA_ML_DL.pdf", poppler_path=poppler_path)
print('conversion done')

#le resultat de convert_from_path est une liste d'image de type pill
print(f"Type: {type(images)}")  # <class 'list'>
print(f"Nombre de pages: {len(images)}")
print(f"Type de chaque image: {type(images[0])}")  # <class 'PIL.Image.Image'>

#avant d'utiliser nos images avec opencv on doit convertir le format pill des images en array
images=np.array(images)
print("conversion Pill en array terminée!")
image=images[0]  # Get the first image from the list

#le nettoyage des images using threshold (threshold transforme une image en noir et blanc pur)=>on passes d’une image en niveaux de gris à une image totalement noire et blanche, très nette
image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
tresh=cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)# cv2.threshold retourne 2 valeurs : retval = seuil utilisé, dst = image binarisée (numpy.ndarray) avec pixels 0 ou 255

print("Nettoyage terminé !")


#utilisation d'un model ia pour faire l'ocr
#u can't send image to gemini or any ai without transform it to byte code
#encoder l'image en PNG
success, buffer = cv2.imencode(".png", tresh[1])
# convertir en bytes
img_bytes = buffer.tobytes()
'''Quand on utilise une API externe, c’est notre application (via un client) qui interroge cette API.
le client envoie des requêtes à une API externe afin d’obtenir des données ou des services.'''
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png",
                        data=img_bytes
                    )
                ),
                types.Part(
                    text="Extract all text from this image"
                )
            ]
        )
    ]
)

text = response.text
print(text)
# RecursiveCharacterTextSplitter splits long texts into smaller overlapping chunks.
# Arguments:
#   chunk_size: Maximum number of characters in each chunk (here, 1200 chars per chunk)
#   chunk_overlap: Number of characters that overlap between adjacent chunks (here, 200 chars overlap)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Each chunk will be up to 1200 characters
    chunk_overlap=50,  # 200 characters of overlap between consecutive chunks
)
chunks =splitter.split_text(response.text)

# Afficher le nombre de chunks
print(f"Nombre de chunks : {len(chunks)}\n")

# Afficher tous les chunks ou juste les premiers pour vérifier
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} ({len(chunk)} caractères) :\n{chunk}\n")

#embiddings
embiddings = []
for chunk in chunks:
    response= client.models.embed_content(
        model="gemini-embedding-001",
        contents=[chunk]   # liste avec 1 élément
    )
    embiddings.append(response.embeddings[0].values)
print(f"Nombre de vecteurs créés : {len(embiddings)}")
print(f"Taille d’un vecteur : {len(embiddings[0])}")


# Initialiser le client Chroma
client_db = chromadb.Client()
collection = client_db.create_collection(name="my_collection")
# Ajouter des chunks et embeddings
collection.add(
    documents=chunks,
    embeddings=embiddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print('collection added')

# Récupérer tous les documents

all_docs = collection.get(include=["documents", "embeddings"])
print(all_docs["documents"])   # tes chunks
print(all_docs["embeddings"])  # tes vecteurs
