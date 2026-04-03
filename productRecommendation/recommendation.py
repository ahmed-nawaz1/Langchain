from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import json
import os

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

vectorstore = PineconeVectorStore(
    index_name="recommendation",
    embedding=embeddings,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)


@tool
def search_similar_products(product_name: str) -> str:
    """Search similar products from vector store based on product name."""
    try:
        results = vectorstore.similarity_search(product_name, k=3)

        products = []
        for doc in results:
            m = doc.metadata

            if m.get("title", "").lower() == product_name.lower():
                continue

            products.append(
                {
                    "id": str(m.get("id", "")),
                    "title": str(m.get("title", "")),
                    "price": float(m.get("price", 0)),
                    "finalPrice": float(m.get("finalPrice", 0)),
                    "discount": float(m.get("discount", 0)),
                    "stock": int(m.get("stock", 0)),
                    "image": str(m.get("image", "")),
                }
            )

        return json.dumps(products, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)})


tools = [search_similar_products]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="""
You are a product recommendation agent.
Use the search_similar_products tool.
Return ONLY valid JSON array.
Each product must include:
id, title, price, finalPrice, discount, stock, image
No explanation.
No markdown.
Just JSON.
""",
)


def get_recommendations(product_name: str):
    try:
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f'User is viewing: "{product_name}". Return JSON array of recommended products with full product data including id, title, price, finalPrice, discount, stock, image.',
                    }
                ]
            }
        )

        output = response["messages"][-1].content

        if isinstance(output, list):
            text_parts = []
            for item in output:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                else:
                    text_parts.append(str(item))
            output = "".join(text_parts)

        if "```" in output:
            output = output.split("```")[1]
            if output.startswith("json"):
                output = output[4:]
            output = output.strip()

        start = output.find("[")
        end = output.rfind("]") + 1

        if start != -1 and end != 0:
            return json.loads(output[start:end])

        return []

    except Exception:
        return []


if __name__ == "__main__":
    product_name = "cars"
    results = get_recommendations(product_name)
    print(json.dumps(results, indent=2, ensure_ascii=False))
