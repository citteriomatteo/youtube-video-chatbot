import spacy

from vector_db.chroma import EnhVectorDatabase


def load_transcript_into_vector_storage(vector_db: EnhVectorDatabase, filename: str = None, transcript: str = None):
    """
    Reads the transcript from the txt file at the path (if given, otherwise it is expected to directly have the text).
    Then, chunkizes the text and loads the chunks into the vector storage (with default encoding).
    """
    if filename is not None:
        with open(filename, 'r', encoding='utf-8') as file:
            transcript = file.read()
    chunks = get_transcript_chunks(transcript=transcript)
    vector_db.insert_new_chunks(documents=chunks, use_default_encoding=True)


def get_transcript_chunks(transcript):
    """
    Uses spaCy library to chunkize the whole transcript.
    """
    nlp = spacy.load('it_core_news_sm')
    doc = nlp(transcript)
    chunks = [sent.text for sent in doc.sents]
    return chunks
