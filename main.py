from vector_db.chroma import EnhVectorDatabase
from video_data_ingestion.pdf_retrieval_ingestion import download_and_load_pdf_from_url
from video_data_ingestion.transcript_retrieval_ingestion import download_and_load_yt_transcript_from_url
from question_answering.question_answering import make_a_question

from config import N_MOST_RELEVANT_CHUNKS, YOUTUBE_URL, PDF_URL


if __name__ == '__main__':

    storage = EnhVectorDatabase()

    download_and_load_yt_transcript_from_url(url=YOUTUBE_URL, vector_db=storage)

    download_and_load_pdf_from_url(url=PDF_URL, vector_db=storage)

    while True:
        question = input("Question: ")
        response = make_a_question(question=question, vector_db=storage, n_relevant_documents=N_MOST_RELEVANT_CHUNKS)
        print("Response: ", response)
