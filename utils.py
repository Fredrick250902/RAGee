import os
import requests
import pypdf
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Router API (MiniLM – 384 dim)
API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

headers = {
    "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
    "Content-Type": "application/json"
}


def extract_chunks(pdf_file):
    """
    Extract text from PDF and split into larger overlapping chunks.
    """
    reader = pypdf.PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # Larger chunks → fewer embeddings
    size = 1200
    overlap = 150

    return [
        text[i : i + size]
        for i in range(0, len(text), size - overlap)
    ]


def get_embedding(text):
    """
    Generate embedding for a single text (used for query).
    """
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text},
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(
            f"HF Request Failed: {response.status_code} - {response.text}"
        )

    embedding = response.json()

    # Flatten if needed
    if isinstance(embedding, list) and isinstance(embedding[0], list):
        embedding = embedding[0]

    return [float(x) for x in embedding]


def get_embeddings(texts):
    """
    Generate embeddings for a batch of texts.
    Returns: List[List[float]]
    """
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": texts},
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(
            f"HF Request Failed: {response.status_code} - {response.text}"
        )

    embeddings = response.json()

    # HF router may return nested lists
    if isinstance(embeddings[0][0], list):
        embeddings = [e[0] for e in embeddings]

    return [[float(x) for x in emb] for emb in embeddings]
