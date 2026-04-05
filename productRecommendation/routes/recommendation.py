from fastapi import  APIRouter , Body 
from controllers.recommendation_controller import recommend_products

router = APIRouter(
    prefix="/api",
)

@router.post("/recommendations")
def get_recommendations(
    product_name : str = Body(..., embed=True)
) :
    return recommend_products(product_name)