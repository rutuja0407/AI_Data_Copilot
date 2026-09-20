import pandas as pd
import pytest

from backend.services.ai import (
    CodePlanPayload,
    PlanPayload,
    PlanValidationError,
)
from backend.services.executor import CodeExecutionError, execute_generated_code


def make_dataframe():
    return pd.DataFrame(
        {
            "age": [25, None, 40, 30],
            "salary": [50000, 52000, None, 61000],
            "country": ["US", "US", "DE", None],
        }
    )


def test_plan_validation_accepts_supported_actions():
    df = make_dataframe()
    plan_dict = {
        "plan_overview": "Fill and clip",
        "confidence": 0.9,
        "steps": [
            {
                "description": "Fill missing age with median",
                "column": "age",
                "action": "fill_missing",
                "parameters": {"method": "median"},
            },
            {
                "description": "Drop duplicate rows",
                "column": None,
                "action": "drop_duplicates",
                "parameters": {},
            },
        ],
    }

    plan = PlanPayload.model_validate(plan_dict)
    validated = plan.validate_against_dataframe(df)
    assert validated.steps[0].parameters["method"] == "median"


def test_plan_validation_blocks_unknown_column():
    df = make_dataframe()
    plan_dict = {
        "steps": [
            {
                "description": "Attempt invalid column",
                "column": "unknown",
                "action": "fill_missing",
                "parameters": {"method": "median"},
            }
        ]
    }

    plan = PlanPayload.model_validate(plan_dict)
    with pytest.raises(PlanValidationError):
        plan.validate_against_dataframe(df)


def test_plan_validation_requires_constant_value():
    df = make_dataframe()
    plan_dict = {
        "steps": [
            {
                "description": "Constant fill must include value",
                "column": "country",
                "action": "fill_missing",
                "parameters": {"method": "constant"},
            }
        ]
    }

    plan = PlanPayload.model_validate(plan_dict)
    with pytest.raises(PlanValidationError):
        plan.validate_against_dataframe(df)


def test_plan_validation_rejects_non_numeric_clip():
    df = make_dataframe()
    plan_dict = {
        "steps": [
            {
                "description": "Cannot clip text column",
                "column": "country",
                "action": "clip_outliers",
                "parameters": {},
            }
        ]
    }

    plan = PlanPayload.model_validate(plan_dict)
    with pytest.raises(PlanValidationError):
        plan.validate_against_dataframe(df)


def test_plan_validation_allows_chart_with_known_columns():
    df = make_dataframe()
    plan_dict = {
        "steps": [
            {
                "description": "Drop duplicates",
                "column": None,
                "action": "drop_duplicates",
                "parameters": {},
            }
        ],
        "chart": {
            "chart_type": "scatter",
            "x": "age",
            "y": "salary",
            "intent": "age vs salary",
            "parameters": {},
        },
    }

    plan = PlanPayload.model_validate(plan_dict)
    validated = plan.validate_against_dataframe(df)
    assert validated.chart.chart_type == "scatter"


def test_plan_validation_rejects_chart_unknown_column():
    df = make_dataframe()
    plan_dict = {
        "steps": [
            {
                "description": "Drop duplicates",
                "column": None,
                "action": "drop_duplicates",
                "parameters": {},
            }
        ],
        "chart": {
            "chart_type": "scatter",
            "x": "unknown",
            "y": "salary",
        },
    }

    plan = PlanPayload.model_validate(plan_dict)
    with pytest.raises(PlanValidationError):
        plan.validate_against_dataframe(df)


def test_code_plan_validation_accepts_known_columns():
    df = make_dataframe()
    plan_dict = {
        "intent_summary": "Fill missing age with median",
        "columns_used": ["age"],
        "actions": ["fillna"],
        "code": """
def transform(df):
    df = df.copy()
    df['age'] = df['age'].fillna(df['age'].median())
    return {'dataframe': df, 'summary': 'Filled missing age with median.'}
""",
    }

    plan = CodePlanPayload.model_validate(plan_dict)
    validated = plan.validate_against_dataframe(df)
    assert validated.columns_used == ["age"]


def test_code_plan_validation_blocks_unknown_column():
    df = make_dataframe()
    plan_dict = {
        "intent_summary": "Reference missing column",
        "columns_used": ["missing"],
        "code": "def transform(df):\n    return {'dataframe': df}"
    }

    plan = CodePlanPayload.model_validate(plan_dict)
    with pytest.raises(PlanValidationError):
        plan.validate_against_dataframe(df)


def test_execute_generated_code_returns_dataframe():
    df = make_dataframe()
    code = """
def transform(df):
    df = df.copy()
    df['age'] = df['age'].fillna(df['age'].median())
    preview = df.head(2).to_dict(orient='records')
    return {
        'dataframe': df,
        'preview': preview,
        'summary': 'Filled missing age with median.'
    }
"""

    result = execute_generated_code(code, df)
    assert "dataframe" in result
    assert isinstance(result["dataframe"], pd.DataFrame)
    assert result["dataframe"]["age"].isna().sum() == 0
    assert result["summary"] == "Filled missing age with median."


def test_execute_generated_code_requires_transform():
    df = make_dataframe()
    code = "def other(df):\n    return {'dataframe': df}"

    with pytest.raises(CodeExecutionError):
        execute_generated_code(code, df)
