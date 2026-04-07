from services.recommendation import get_recommendations

def recommend_products(product_name: str):
    try:
        print(f"Received product name for recommendation: '{product_name}'")
        results = get_recommendations(product_name)
        print(f"Recommendations  {results}")
        return {
            "status": 200,
            "success": True,
            "message": "Recommendations fetched successfully",
            "recommendations": results,
        }
    except Exception as e:
        print(f"[ERROR] recommend_products: {str(e)}")

        return {
            "status": 500,
            "success": False,
            "message": "Internal Server Error",
            "error": str(e),
        }
