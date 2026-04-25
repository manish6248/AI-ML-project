from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd

app = FastAPI(
    title="Employee Retention Prediction API",
    description="Predicts whether an employee is likely to leave the company.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("employee_retention_model.pkl")

FEATURE_ORDER = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "low",
    "medium",
]


class Employee(BaseModel):
    satisfaction_level: float = Field(..., ge=0.0, le=1.0, example=0.38)
    last_evaluation: float = Field(..., ge=0.0, le=1.0, example=0.53)
    number_project: int = Field(..., ge=1, example=2)
    average_montly_hours: int = Field(..., ge=1, example=157)
    time_spend_company: int = Field(..., ge=0, example=3)
    Work_accident: Literal[0, 1] = Field(..., example=0)
    promotion_last_5years: Literal[0, 1] = Field(..., example=0)
    salary: Literal["low", "medium", "high"] = Field(..., example="low")


class Prediction(BaseModel):
    will_leave: bool
    leave_probability: float
    stay_probability: float


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict(emp: Employee):
    try:
        row = {
            "satisfaction_level": emp.satisfaction_level,
            "last_evaluation": emp.last_evaluation,
            "number_project": emp.number_project,
            "average_montly_hours": emp.average_montly_hours,
            "time_spend_company": emp.time_spend_company,
            "Work_accident": emp.Work_accident,
            "promotion_last_5years": emp.promotion_last_5years,
            "low": emp.salary == "low",
            "medium": emp.salary == "medium",
        }
        X = pd.DataFrame([row], columns=FEATURE_ORDER)

        proba = model.predict_proba(X)[0]
        stay_prob, leave_prob = float(proba[0]), float(proba[1])

        return Prediction(
            will_leave=leave_prob >= 0.5,
            leave_probability=round(leave_prob, 4),
            stay_probability=round(stay_prob, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
