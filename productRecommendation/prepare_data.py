from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from retrive_data import get_mongo_client
import os
import time

load_dotenv()

docs = get_mongo_client()

if not docs:
    raise ValueError("No documents found from MongoDB")

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in .env file")

pc = Pinecone(api_key=pinecone_api_key)

def create_pinecone_index():
    index_name = "recommendation"

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        time.sleep(5)

    vector_store = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embedding,
        index_name=index_name
    )

    print("Documents successfully converted into vectors and stored in Pinecone.")

    query = "watches"
    results = vector_store.similarity_search(query, k=3)

    for i, res_doc in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print("Content:", res_doc.page_content[:300])
        print("Metadata:", res_doc.metadata)

if __name__ == "__main__":
    create_pinecone_index()