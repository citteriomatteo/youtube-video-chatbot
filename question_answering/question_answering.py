import openai
from vector_db.chroma import EnhVectorDatabase
from dotenv import load_dotenv
import os

# Carica le variabili di ambiente dal file .env
load_dotenv()


def make_a_question(question: str, vector_db: EnhVectorDatabase, n_relevant_documents=1):

    most_similar_chunks = vector_db.get_most_similar_documents(
        collection_name="embeddings",
        query_text=question,
        n=n_relevant_documents,
        use_default_encoding=True
    )
    prompt_content = "\n".join(most_similar_chunks)

    prompt = f"You are an agent that must respond to questions regarding a youtube video and some text coming from PDFs. " \
             f"These are the chunks of text: \n {prompt_content} \n" \
             f"Answer the question sticking to the data provided above."
    # prompt = f"Sei un agente che deve rispondere a domande riguardanti un video youtube. Questo è il transcript del video in questione: {prompt_content}\nRispondi alle domande che ti vengono poste attinendoti al transcript."

    # use gpt-3.5-turbo in a RAG fashion to obtain a response
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        stream=True,
    )

    text_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            text_response += chunk.choices[0].delta.content

    return text_response
