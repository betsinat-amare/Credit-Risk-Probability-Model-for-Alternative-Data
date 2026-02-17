from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    TotalTransactionAmount: float = Field(..., description="Sum of all transaction amounts for the customer", example=1500.50)
    AvgTransactionAmount: float = Field(..., description="Average transaction amount", example=150.05)
    TransactionCount: int = Field(..., description="Total number of transactions", example=10)
    StdTransactionAmount: Optional[float] = Field(0.0, description="Standard deviation of transaction amounts", example=50.2)


class PredictionResponse(BaseModel):
    risk_probability: float = Field(..., description="Probability of being a high-risk customer (0 to 1)", example=0.15)
    risk_category: str = Field(..., description="Risk category (Low, Medium, High)", example="Low")
