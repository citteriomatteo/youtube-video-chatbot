import fitz
import requests
from io import BytesIO

from tqdm import tqdm

from vector_db.chroma import EnhVectorDatabase
from video_data_ingestion.ingestion_utils import load_transcript_into_vector_storage


def download_and_load_pdf_from_url(vector_db: EnhVectorDatabase, url: str):
    """
    Manages to retrieve the text related to the requested pdf and load it into the vector storage.
    """
    print("PDF: downloading transcript...")
    transcript = get_pdf_transcript_from_url(url=url)

    print("PDF: loading transcript into vector storage...")
    load_transcript_into_vector_storage(vector_db=vector_db, transcript=transcript)


def get_pdf_transcript_from_url(url: str):
    """
    Retrieves the transcript from the pdf's url.
    """

    response = requests.get(url)
    response.raise_for_status()

    pdf_document = fitz.open(stream=BytesIO(response.content))

    text = ""
    for page_number in tqdm(range(pdf_document.page_count), desc="PDF text extraction"):
        page = pdf_document[page_number]
        text += page.get_text()

    pdf_document.close()

    return text
