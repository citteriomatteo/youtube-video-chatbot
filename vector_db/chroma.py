import chromadb
from sentence_transformers import SentenceTransformer

from config import DEF_COLLECTION_NAME, USE_DEF_ENCODING, N_MOST_RELEVANT_CHUNKS


class EnhVectorDatabase:
    """
    Class to enhance the basic Chroma features with a custom encoding strategy (using sentence-BERT).
    """

    def __init__(self, collection_name: str = None):
        """
        Initializes Chroma storage, the default collection and the custom encoder.
        :param collection_name: optional collection name (otherwise, the default is used)
        """

        if collection_name is None:
            collection_name = DEF_COLLECTION_NAME

        self.chroma_client = chromadb.Client()
        self.custom_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.collections = {
            collection_name: self.chroma_client.create_collection(name=collection_name)
        }

    def create_new_collection(self, collection_name: str):
        """
        Creates a new custom collection, if needed.
        :param collection_name: the collection name
        """
        if collection_name in self.collections:
            raise Exception(f"Already existing collection with name {collection_name}.")
        self.collections.update(
            {collection_name: self.chroma_client.create_collection(name=collection_name)}
        )

    def insert_new_chunks(self, documents: [], use_default_encoding: bool = USE_DEF_ENCODING, collection_name: str = None):
        """
        Inserts, in the requested collection, the given documents using default or custom encoding.
        :param documents: list of texts
        :param use_default_encoding: bool to choose encoding strategy
        :param collection_name: the collection name
        """
        if collection_name is None:
            collection_name = DEF_COLLECTION_NAME

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

    def get_most_similar_documents(self, query_text: str, n=N_MOST_RELEVANT_CHUNKS, use_default_encoding: bool = USE_DEF_ENCODING, collection_name: str = DEF_COLLECTION_NAME):
        """
        Returns the "n" most similar text chunks related to textual query.
        :param query_text: textual query
        :param n: optional number of most relevant chunks to consider
        :param use_default_encoding: bool to choose between default and custom encoding strategy
        :param collection_name: optional collection name
        :return: list of "n" most relevant texts
        """
        if collection_name is None:
            collection_name = DEF_COLLECTION_NAME

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
