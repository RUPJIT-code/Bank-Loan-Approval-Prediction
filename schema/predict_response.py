from pydantic import BaseModel, Field
from typing import Dict, Literal


class PredictionResponse(BaseModel):

    prediction: Literal["Loan Approved", "Loan Rejected"] = Field(...,
        description="Final loan approval prediction.")

    confidence: float = Field(...,ge=0,le=1,
        description="Confidence score of the predicted outcome.")

    probabilities: Dict[str, float] = Field(...,
        description="Probability distribution for each possible outcome.",
        example={"Loan Rejected": 0.1567,"Loan Approved": 0.8433})