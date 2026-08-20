from pydantic import BaseModel,Field,computed_field
from typing import Annotated,Literal

class UserInput(BaseModel):
    Gender:Annotated[Literal["Male", "Female"],Field(...,description='Gender')]
    married:Annotated[Literal["Yes", "No"],Field(...,description='Marital Status')]
    dependents:Annotated[Literal["0", "1", "2", "3+"],Field(...,description='Financially Dependent (e.g., children, spouse, or parents)?')]
    education:Annotated[Literal["Graduated", "Not Graduated"],Field(...,description='Education Status')]
    self_employed:Annotated[Literal["No", "Yes"],Field(...,description='Self-Employed')]
    property_area:Annotated[Literal["Urban", "Semiurban", "Rural"],Field(...,description='Property Area')]
    credit_history:Annotated[Literal["Has credit history (1)", "No credit history (0)"],Field(...,description='Credit history')]
    applicant_income:Annotated[float,Field(...,description='Applicant Monthly Income ($)',ge=0)]
    loan_amount:Annotated[float,Field(...,description='Loan Amount (in thousands, e.g. 128 = $128,000)',ge=0)]
    coapplicant_income:Annotated[float,Field(...,description='Coapplicant Monthly Income ($)')]
    loan_term_years: Annotated[int,Field(..., description="Loan Term (Years)", ge=0,le=20)]
    loan_term_months: Annotated[int,Field(..., description="Additional Months", ge=0, le=11)]

    @computed_field
    @property
    def loan_amount_term(self) -> int:
        """Loan term in days (used by the ML model)."""
        total_months = (
            self.loan_term_years * 12
            + self.loan_term_months
        )
        return total_months * 30