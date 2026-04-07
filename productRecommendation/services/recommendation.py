from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import json
import os
import time
import traceback

load_dotenv()

print("\n[STEP 1] Loading embeddings model...")
t0 = time.perf_counter()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

print(f"[OK] Embeddings initialized in {time.perf_counter() - t0:.2f}s")

print("\n[STEP 2] Connecting Pinecone...")
t1 = time.perf_counter()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

vectorstore = PineconeVectorStore(
    index_name="recommendation",
    embedding=embeddings,
)

print(f"[OK] Pinecone connected in {time.perf_counter() - t1:.2f}s")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)

print("\n[STEP 3] Initializing Hugging Face LLM...")
t2 = time.perf_counter()

# repo_id = "HuggingFaceH4/zephyr-7b-beta"

# endpoint = HuggingFaceEndpoint(
#     repo_id=repo_id,
#     temperature=0.2,
#     max_new_tokens=300,
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
#     provider="auto",
# )

# llm = ChatHuggingFace(llm=endpoint)

print(f"[OK] LLM initialized in {time.perf_counter() - t2:.2f}s")


@tool
def search_similar_products(product_name: str) -> str:
    """Search similar products from vector store based on product name."""
    try:
        print("\n[TOOL] search_similar_products called")
        print(f"[TOOL] Query: {product_name}")

        t_tool = time.perf_counter()
        results = vectorstore.similarity_search(product_name, k=5)
        print(f"[TOOL] similarity_search completed in {time.perf_counter() - t_tool:.2f}s")
        print(f"[TOOL] Raw docs found: {len(results)}")

        products = []
        for i, doc in enumerate(results, start=1):
            m = doc.metadata
            print(f"[TOOL] Doc {i} metadata: {m}")

            title = str(m.get("title", "")).strip()

            if title.lower() == product_name.lower():
                print(f"[TOOL] Skipped exact same title: {title}")
                continue

            products.append(
                {
                    "id": str(m.get("id", "")),
                    "title": title,
                    "price": float(m.get("price", 0) or 0),
                    "finalPrice": float(m.get("finalPrice", 0) or 0),
                    "discount": float(m.get("discount", 0) or 0),
                    "stock": int(m.get("stock", 0) or 0),
                    "image": str(m.get("image", "")),
                }
            )

        print(f"[TOOL] Final filtered products count: {len(products)}")
        print(f"[TOOL] Final products JSON: {json.dumps(products, ensure_ascii=False, indent=2)}")

        return json.dumps(products, ensure_ascii=False)

    except Exception as e:
        print(f"[TOOL ERROR] {str(e)}")
        traceback.print_exc()
        return json.dumps({"error": str(e)})


tools = [search_similar_products]

print("\n[STEP 4] Creating agent...")
t3 = time.perf_counter()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt = """
You are a product recommendation agent.

IMPORTANT:
- You MUST call the search_similar_products tool.
- You MUST return EXACTLY the same output from the tool.
- DO NOT filter, modify, or remove any product.
- DO NOT judge relevance.
- Even if products seem unrelated, return them.

Output:
- Return ONLY JSON array
- No explanation
- No changes
""" , 
)

print(f"[OK] Agent created in {time.perf_counter() - t3:.2f}s")


def get_recommendations(product_name: str):
    try:
        print("\n==================== GET RECOMMENDATIONS START ====================")
        print(f"[INPUT] product_name: {product_name}")

        t_main = time.perf_counter()

        print("\n[STEP 5] Invoking agent...")
        t_invoke = time.perf_counter()

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f'''
User is viewing: "{product_name}"

Find and return only highly relevant products related to this product name.
Priority:
1. Same product type
2. Same category
3. Same niche
4. Closely related alternatives

Use the search_similar_products tool.
Do not return unrelated or random products.
If nothing relevant is found, return [].

Return JSON array only with:
id, title, price, finalPrice, discount, stock, image
''',
                    }
                ]
            }
        )

        invoke_time = time.perf_counter() - t_invoke
        print(f"[STEP 5 DONE] Agent invoke completed in {invoke_time:.2f}s")

        print("\n[DEBUG] Full agent response object:")
        print(response)

        messages = response.get("messages", [])
        print(f"\n[DEBUG] Total messages returned: {len(messages)}")

        if not messages:
            print("[WARN] No messages returned from agent")
            return []

        last_message = messages[-1]
        print("\n[DEBUG] Last message object:")
        print(last_message)

        output = getattr(last_message, "content", last_message)
        print("\n[DEBUG] Raw output before normalization:")
        print(output)

        if isinstance(output, list):
            text_parts = []
            for item in output:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                else:
                    text_parts.append(str(item))
            output = "".join(text_parts)

        output = str(output).strip()

        print("\n[DEBUG] Output after normalization:")
        print(output)

        if "```" in output:
            parts = output.split("```")
            if len(parts) > 1:
                output = parts[1]
                if output.startswith("json"):
                    output = output[4:]
                output = output.strip()

        print("\n[DEBUG] Output after markdown cleanup:")
        print(output)

        start = output.find("[")
        end = output.rfind("]") + 1

        print(f"\n[DEBUG] JSON bracket positions => start: {start}, end: {end}")

        if start != -1 and end != 0 and end > start:
            parsed = json.loads(output[start:end])
            print(f"[SUCCESS] Parsed JSON items count: {len(parsed)}")
            print(f"[TOTAL TIME] {time.perf_counter() - t_main:.2f}s")
            print("==================== GET RECOMMENDATIONS END ====================\n")
            return parsed

        print("[WARN] Could not find valid JSON array in model output")
        print(f"[TOTAL TIME] {time.perf_counter() - t_main:.2f}s")
        print("==================== GET RECOMMENDATIONS END ====================\n")
        return []

    except Exception as e:
        print(f"[ERROR] get_recommendations: {str(e)}")
        traceback.print_exc()
        return []


if __name__ == "__main__":
    product_name = "cars"
    results = get_recommendations(product_name)
    print("\n==================== FINAL RESULT ====================")
    print(json.dumps(results, indent=2, ensure_ascii=False))