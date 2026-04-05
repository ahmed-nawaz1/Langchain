from dotenv import load_dotenv
from langchain_core.documents import Document
from pymongo import MongoClient
import os

load_dotenv()


def get_mongo_client():
    mongo_url = os.getenv("MONGO_DB")

    if not mongo_url:
        raise ValueError("MONGO_DB not found in .env file")

    client = MongoClient(mongo_url)
    db = client["FYP"]
    collection = db["sellerproducts"]

    data = list(collection.find())

    docs = []

    for p in data:
        content = f"""
        Title: {p.get('title', '')}
        Description: {p.get('description', '')}
        Price: {p.get('finalPrice', '')} PKR
        Discount: {p.get('discount', '')}%
        Stock: {p.get('stock', '')} units available
        """.strip()

        doc = Document(
            page_content=content,
            metadata={
                "id": str(p.get("_id", "")),
                "title": p.get("title", ""),
                "price": p.get("price", 0),
                "finalPrice": p.get("finalPrice", 0),
                "discount": p.get("discount", 0),
                "stock": p.get("stock", 0),
                "image": p.get("image", [""])[0] if p.get("image") else "",
                "categoryId": str(p.get("categoryId", "")),
                "sellerId": str(p.get("sellerId", ""))
            }
        )

        docs.append(doc)

    print(f"Total Documents created: {len(docs)}")
    return docs


if __name__ == "__main__":
    documents = get_mongo_client()
    print(documents[:2])