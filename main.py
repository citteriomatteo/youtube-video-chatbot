from vector_db.chroma import EnhVectorDatabase
from video_data_ingestion.pdf_retrieval_ingestion import download_and_load_pdf_from_url
from video_data_ingestion.transcript_retrieval_ingestion import download_and_load_yt_transcript_from_url
from question_answering.question_answering import make_a_question

if __name__ == '__main__':

    storage = EnhVectorDatabase(collection_name="embeddings")

    url_video_youtube = 'https://www.youtube.com/watch?v=kqtD5dpn9C8'
    download_and_load_yt_transcript_from_url(url=url_video_youtube, vector_db=storage)

    url_pdf = "https://www.tutorialspoint.com/python/pdf/python_quick_guide.pdf"
    download_and_load_pdf_from_url(url=url_pdf, vector_db=storage)

    while True:
        question = input("Question: ")
        response = make_a_question(question=question, vector_db=storage, n_relevant_documents=4)
        print("Response: ", response)
