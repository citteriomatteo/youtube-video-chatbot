import chromadb
from sentence_transformers import SentenceTransformer


class EnhVectorDatabase:

    def __init__(self, collection_name: str = "embeddings"):
        self.chroma_client = chromadb.Client()
        self.custom_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.collections = {
            collection_name: self.chroma_client.create_collection(name=collection_name)
        }

    def create_new_collection(self, collection_name: str):
        if collection_name in self.collections:
            raise Exception(f"Already existing collection with name {collection_name}.")
        self.collections.update(
            {collection_name: self.chroma_client.create_collection(name=collection_name)}
        )

    def insert_new_chunks(self, collection_name: str, documents: [], use_default_encoding: bool = True):

        collection = self.collections[collection_name]

        if not use_default_encoding:  # using custom encoding with sBERT
            embeddings = self.custom_encoder.encode(documents)
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=[{"source": "test"} for _ in range(len(documents))],
                ids=[str(collection.count() + i) for i in range(len(documents))]
            )

        else:  # using default chroma encoding
            collection.add(
                documents=documents,
                metadatas=[{"source": "test"} for _ in range(len(documents))],
                ids=[str(collection.count() + i) for i in range(len(documents))]
            )

    def get_most_similar_documents(self, collection_name: str, query_text: str, n=1, use_default_encoding=True):
        # obtain "query_embedding" by using sBERT
        collection = self.collections[collection_name]

        if use_default_encoding:
            return collection.query(
                query_texts=[query_text],
                n_results=n
            )["documents"][0]

        query_embedding = self.custom_encoder.encode([query_text])[0].tolist()
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n
        )["documents"][0]
