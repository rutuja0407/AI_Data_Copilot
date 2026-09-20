import json
import logging
import math
import os
import re
import textwrap
import uuid
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, Field, ValidationError, field_validator

try:
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency during local dev/tests
    genai = None

logger = logging.getLogger(__name__)

ALLOWED_ACTIONS = {
    "fill_missing",
    "drop_duplicates",
    "drop_rows_with_missing",
    "clip_outliers",
}

NUMERIC_IMPUTE_METHODS = {"mean", "median"}
SUPPORTED_FILL_METHODS = NUMERIC_IMPUTE_METHODS | {"mode", "constant"}
DEFAULT_CHART_TYPES = {"histogram", "scatter", "bar", "line", "box"}


class ChartPlanPayload(BaseModel):
    intent_summary: str = Field(..., min_length=1)
    chart_type: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    facet_col: Optional[str] = None
    facet_row: Optional[str] = None
    aggregations: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    code: str = Field(..., min_length=1)
    warnings: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None

    @field_validator("aggregations", "warnings", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if item is None:
                    continue
                cleaned.append(str(item))
            return cleaned
        return [str(value)]

    def validate_against_dataframe(self, df: pd.DataFrame) -> "ChartPlanPayload":
        columns = set(df.columns)
        parameters = self.parameters or {}

        def _normalize_to_set(value: Any) -> set[str]:
            if value is None:
                return set()
            if isinstance(value, str):
                return {value}
            if isinstance(value, (list, tuple, set)):
                return {str(item) for item in value if item is not None}
            if isinstance(value, dict):
                normalized = set()
                for key, item in value.items():
                    if key is not None:
                        normalized.add(str(key))
                    if isinstance(item, str):
                        normalized.add(item)
                    elif isinstance(item, (list, tuple, set)):
                        normalized.update({str(sub) for sub in item if sub is not None})
                return normalized
            return {str(value)}

        derived_columns = _normalize_to_set(parameters.get("derived_columns"))
        synthetic_columns = _normalize_to_set(parameters.get("synthetic_columns"))
        result_columns = _normalize_to_set(parameters.get("result_columns"))
        output_columns = _normalize_to_set(parameters.get("output_columns"))

        for label, value in {
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "facet_col": self.facet_col,
            "facet_row": self.facet_row,
        }.items():
            if value is not None and value not in columns:
                if value in derived_columns or value in synthetic_columns or value in result_columns or value in output_columns:
                    continue
                raise PlanValidationError(f"Chart field '{label}' references unknown column '{value}'.")

        if self.chart_type and self.chart_type not in DEFAULT_CHART_TYPES:
            raise PlanValidationError(
                f"Unsupported chart_type '{self.chart_type}'. Allowed: {sorted(DEFAULT_CHART_TYPES)} or omit it."
            )

        if "transform" not in self.code:
            raise PlanValidationError("Generated chart code must define transform(df) function.")

        return self


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return sanitize_for_json(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return sanitize_for_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, np.int64, np.int32, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if value is None:
        return None
    if isinstance(value, (np.ndarray,)):
        return sanitize_for_json(value.tolist())
    return value


class PlanValidationError(Exception):
    """Raised when the LLM produces an invalid plan."""


class PlanStep(BaseModel):
    description: str = Field(..., min_length=1)
    column: Optional[str] = None
    action: str = Field(..., min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None

    def validate_against_dataframe(self, df: pd.DataFrame) -> None:
        if isinstance(self.column, str) and self.column.strip().lower() in {"", "none", "null"}:
            self.column = None

        columns = set(df.columns)

        if self.action not in ALLOWED_ACTIONS:
            raise PlanValidationError(f"Unsupported action '{self.action}'.")

        if self.column is not None and self.column not in columns:
            raise PlanValidationError(
                f"Step '{self.description}' references unknown column '{self.column}'."
            )

        if self.action == "fill_missing":
            method = self.parameters.get("method", "mean")
            if method not in SUPPORTED_FILL_METHODS:
                raise PlanValidationError(
                    f"fill_missing supports methods {sorted(SUPPORTED_FILL_METHODS)}, received '{method}'."
                )
            if method in NUMERIC_IMPUTE_METHODS:
                if self.column is None:
                    raise PlanValidationError("Numeric fill_missing steps must reference a column.")
                if not is_numeric_dtype(df[self.column]):
                    raise PlanValidationError(
                        f"Method '{method}' only works on numeric columns. '{self.column}' is {df[self.column].dtype}."
                    )
            if method == "constant" and "value" not in self.parameters:
                raise PlanValidationError("fill_missing with method 'constant' requires a 'value' parameter.")

        if self.action == "clip_outliers":
            if self.column is None:
                raise PlanValidationError("clip_outliers requires a column name.")
            if not is_numeric_dtype(df[self.column]):
                raise PlanValidationError(
                    f"clip_outliers only supports numeric columns. '{self.column}' is {df[self.column].dtype}."
                )

        if self.action == "drop_rows_with_missing":
            subset = self.parameters.get("subset")
            if subset is not None:
                if not isinstance(subset, list) or not all(isinstance(item, str) for item in subset):
                    raise PlanValidationError("drop_rows_with_missing subset must be a list of column names.")
                unknown = [col for col in subset if col not in columns]
                if unknown:
                    raise PlanValidationError(
                        f"drop_rows_with_missing subset contains unknown columns: {unknown}."
                    )


class ChartSuggestion(BaseModel):
    intent: Optional[str] = None
    chart_type: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None

    def validate_against_dataframe(self, df: pd.DataFrame) -> None:
        if isinstance(self.chart_type, str) and self.chart_type.strip().lower() in {"", "none", "null"}:
            self.chart_type = None
        if isinstance(self.x, str) and self.x.strip().lower() in {"", "none", "null"}:
            self.x = None
        if isinstance(self.y, str) and self.y.strip().lower() in {"", "none", "null"}:
            self.y = None
        if isinstance(self.color, str) and self.color.strip().lower() in {"", "none", "null"}:
            self.color = None

        if self.chart_type and self.chart_type not in DEFAULT_CHART_TYPES:
            raise PlanValidationError(
                f"Unsupported chart_type '{self.chart_type}'. Allowed: {sorted(DEFAULT_CHART_TYPES)}."
            )

        columns = set(df.columns)
        for label, value in {"x": self.x, "y": self.y, "color": self.color}.items():
            if value is not None and value not in columns:
                raise PlanValidationError(f"Chart field '{label}' references unknown column '{value}'.")


class PlanPayload(BaseModel):
    plan_overview: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    steps: List[PlanStep] = Field(default_factory=list)
    chart: Optional[ChartSuggestion] = None

    def validate_against_dataframe(self, df: pd.DataFrame) -> "PlanPayload":
        if not self.steps:
            if (self.confidence or 0.0) > 0:
                raise PlanValidationError("Plan with non-zero confidence must include at least one step.")
            return self

        for step in self.steps:
            step.validate_against_dataframe(df)

        if self.chart:
            self.chart.validate_against_dataframe(df)

        return self


class CodePlanPayload(BaseModel):
    intent_summary: str = Field(..., min_length=1)
    code: str = Field(..., min_length=1)
    columns_used: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    tests: List[str] = Field(default_factory=list)

    @field_validator("columns_used", "actions", "warnings", "tests", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            flattened = []
            for item in value:
                if item is None:
                    continue
                flattened.append(str(item))
            return flattened
        return [str(value)]

    def validate_against_dataframe(self, df: pd.DataFrame) -> "CodePlanPayload":
        missing = [col for col in self.columns_used if col not in df.columns]
        if missing:
            raise PlanValidationError(
                f"Generated code references unknown columns: {missing}."
            )
        if "transform" not in self.code:
            raise PlanValidationError("Generated code must define transform(df) function.")
        return self


def profile_dataframe(df: pd.DataFrame, sample_size: int = 3) -> Dict[str, Any]:
    """Return a lightweight profile that is safe to send to the LLM."""
    profile: Dict[str, Any] = {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_percent": df.isna().mean().round(4).to_dict(),
        "n_unique": df.nunique(dropna=True).to_dict(),
    }

    try:
        describe_df = df.describe(include="all").transpose().round(4).fillna("null")
        profile["describe"] = describe_df.to_dict(orient="index")
    except Exception as err:  # pragma: no cover - describe can fail for mixed dtype frames
        logger.debug("describe() failed: %s", err)

    head_rows = df.head(sample_size)
    sanitized = head_rows.replace([np.inf, -np.inf], pd.NA)
    sanitized = sanitized.where(pd.notna(sanitized), None)
    profile["sample_rows"] = sanitized.to_dict(orient="records")
    profile = sanitize_for_json(profile)
    return profile


def _ensure_gemini_model() -> Any:
    if genai is None:
        raise RuntimeError("google-generativeai package is not installed. Run `pip install google-generativeai`." )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API key not configured. Set GEMINI_API_KEY environment variable.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest")
    if not hasattr(_ensure_gemini_model, "_model"):
        genai.configure(api_key=api_key)
        _ensure_gemini_model._model = genai.GenerativeModel(model_name)
    return _ensure_gemini_model._model


def _invoke_gemini(prompt: str, *, max_output_tokens: Optional[int] = None) -> str:
    model = _ensure_gemini_model()

    logger.debug("Sending prompt to Gemini (%s chars)", len(prompt))

    configured_max_tokens = max_output_tokens
    if configured_max_tokens is None:
        try:
            configured_max_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
        except ValueError:
            configured_max_tokens = 4096

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
                "top_p": 0.8,
                "max_output_tokens": configured_max_tokens,
                "response_mime_type": "application/json",
            },
        )
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("Gemini API call failed: %s", exc)
        raise RuntimeError("Gemini API call failed") from exc

    raw_text: Optional[str] = None
    try:
        raw_text = response.text  # type: ignore[attr-defined]
    except ValueError as exc:
        logger.warning("Gemini response missing text: %s", exc)
    except AttributeError:
        raw_text = None

    if not raw_text and hasattr(response, "candidates"):
        candidates = getattr(response, "candidates", [])
        gathered_parts: List[str] = []
        finish_reasons: List[Any] = []
        for candidate in candidates:  # pragma: no cover - depends on API response
            finish_reason = getattr(candidate, "finish_reason", None) or getattr(candidate, "finishReason", None)
            if finish_reason is not None:
                finish_reasons.append(finish_reason)
            content = getattr(candidate, "content", None)
            part_iter = []
            if content is None:
                part_iter = []
            elif hasattr(content, "parts"):
                part_iter = content.parts  # type: ignore[attr-defined]
            elif isinstance(content, list):
                part_iter = content
            for part in part_iter:
                text_value = getattr(part, "text", None)
                if text_value:
                    gathered_parts.append(text_value)
        if gathered_parts:
            raw_text = "\n".join(gathered_parts).strip()
        else:
            if finish_reasons:
                raise PlanValidationError(
                    f"Gemini did not return content (finish reasons: {finish_reasons})."
                )
            raise PlanValidationError("Gemini did not return any content for the request.")

    if not raw_text:
        raise PlanValidationError("Gemini returned empty response.")

    logger.debug("Gemini raw response: %s", raw_text)
    return raw_text


def _build_plan_schema_json() -> str:
    schema = {
        "plan_overview": "<short summary>",
        "confidence": "<float between 0 and 1>",
        "steps": [
            {
                "description": "<human readable summary>",
                "column": "<column name or null>",
                "action": "fill_missing | drop_duplicates | drop_rows_with_missing | clip_outliers",
                "parameters": {"method": "mean | median | mode | constant", "value": 0},
                "rationale": "<brief reason>"
            }
        ],
        "chart": {
            "intent": "<what the user wants to see>",
            "chart_type": "histogram | scatter | bar | line | box",
            "x": "<column name or null>",
            "y": "<column name or null>",
            "color": "<column name or null>",
            "parameters": {},
            "rationale": "<brief reason>"
        }
    }
    return json.dumps(schema, indent=2)


def _build_plan_prompt(query: str, profile: Dict[str, Any]) -> str:
    schema = _build_plan_schema_json()
    few_shot = textwrap.dedent(
        """
        Example output for request "fill missing ages and drop duplicates":
        {
          "plan_overview": "Fill missing values and remove duplicates",
          "confidence": 0.82,
          "steps": [
            {
              "description": "Fill missing values in 'age' with the median",
              "column": "age",
              "action": "fill_missing",
              "parameters": {"method": "median"},
              "rationale": "Median works well for skewed numeric columns and preserves distribution."
            },
            {
              "description": "Drop duplicate rows",
              "column": null,
              "action": "drop_duplicates",
              "parameters": {},
              "rationale": "Remove exact duplicate records."
            }
          ]
        }

                Example output for request "fill missing release_date with mode":
                {
                    "plan_overview": "Fill missing release dates",
                    "confidence": 0.75,
                    "steps": [
                        {
                            "description": "Fill missing values in 'release_date' using the mode",
                            "column": "release_date",
                            "action": "fill_missing",
                            "parameters": {"method": "mode"},
                            "rationale": "Mode is valid for categorical/text columns."
                        }
                    ]
                }
        """
    ).strip()

    profile_json = json.dumps(profile, indent=2)
    instructions = textwrap.dedent(
        f"""
        You are a data cleaning assistant. Translate the user's request into a strict JSON plan.
        Use ONLY the actions listed in the schema.
                The action fill_missing supports methods mean, median, mode, and constant.
        If the request is impossible with available actions, return an empty steps array and set confidence to 0.0.
        You MUST output JSON that matches the schema exactly. Do not wrap in backticks or add commentary.

        Dataset profile:
        {profile_json}

        User request:
        {query}

        Schema:
        {schema}

        {few_shot}
        """
    ).strip()

    return instructions


def _extract_json_payload(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3].strip()

    # Attempt to locate the outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise PlanValidationError("LLM response did not contain JSON object.")

    cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on model output
        raise PlanValidationError(f"Failed to parse LLM JSON: {exc}") from exc


def generate_plan_with_gemini(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Call Gemini, validate the plan, and return structured payload for the API."""

    profile = profile_dataframe(df)
    prompt = _build_plan_prompt(query=query, profile=profile)
    raw_text = _invoke_gemini(prompt)

    payload_dict = _extract_json_payload(raw_text)

    try:
        plan = PlanPayload.model_validate(payload_dict)
    except ValidationError as exc:
        raise PlanValidationError(f"Plan schema validation failed: {exc}") from exc

    plan.validate_against_dataframe(df)

    plan_dict = plan.model_dump(exclude_none=True)
    plan_id = str(uuid.uuid4())

    return {
        "plan": plan_dict,
        "plan_id": plan_id,
        "profile": profile,
        "raw_response": raw_text,
    }


def _build_code_schema_json() -> str:
    schema = {
        "intent_summary": "<short restatement>",
        "columns_used": ["<column names involved>"] ,
        "actions": ["<list of actions or pandas operations>"] ,
        "code": "def transform(df):\n    ...\n    return {'dataframe': df} ",
        "explanation": "<rationale>",
        "warnings": ["<optional warnings>"] ,
        "tests": ["<optional quick checks>"]
    }
    return json.dumps(schema, indent=2)


def _build_code_prompt(query: str, profile: Dict[str, Any]) -> str:
    schema = _build_code_schema_json()
    profile_json = json.dumps(profile, indent=2)

    example = textwrap.dedent(
        """
        Example output for request "compute average revenue by country":
        {
          "intent_summary": "Group by country and compute mean revenue",
          "columns_used": ["country", "revenue"],
          "actions": ["groupby", "agg"] ,
          "code": "def transform(df):\n    grouped = df.groupby('country', dropna=False)['revenue'].mean().reset_index(name='avg_revenue')\n    return {\n        'dataframe': grouped,\n        'summary': 'Computed average revenue per country.',\n        'preview': grouped.head(5).to_dict(orient='records')\n    }",
          "explanation": "Group data to derive mean revenue per country.",
          "warnings": [],
          "tests": ["assert 'avg_revenue' in result['dataframe'].columns"]
        }
        """
    ).strip()

    instructions = textwrap.dedent(
        f"""
        You are an expert data scientist. Translate the user's request into executable Python code.
        Use pandas and numpy only. Never import additional libraries.
        You MUST output JSON matching the schema provided. Do not wrap in backticks.
        The code must define a function `transform(df)` that returns a dict containing at least a 'dataframe' key.
        Use df.copy() internally when needed and avoid modifying global state.
        Include the columns and pandas operations you relied on in the respective arrays.
        If the request cannot be satisfied, return an empty actions array, an empty columns_used array, and set code to a transform that simply returns {{'dataframe': df}} with an explanation describing the limitation.

        Dataset profile:
        {profile_json}

        User request:
        {query}

        Schema:
        {schema}

        {example}
        """
    ).strip()
    return instructions


def generate_code_with_gemini(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    profile = profile_dataframe(df)
    prompt = _build_code_prompt(query=query, profile=profile)
    raw_text = _invoke_gemini(prompt)

    payload_dict = _extract_json_payload(raw_text)

    try:
        code_plan = CodePlanPayload.model_validate(payload_dict)
    except ValidationError as exc:
        raise PlanValidationError(f"Code generation schema validation failed: {exc}") from exc

    code_plan.validate_against_dataframe(df)

    plan_dict = code_plan.model_dump(exclude_none=True)
    plan_id = str(uuid.uuid4())

    return {
        "plan": plan_dict,
        "plan_id": plan_id,
        "profile": profile,
        "raw_response": raw_text,
    }


def _build_chart_schema_json() -> str:
    schema = {
        "intent_summary": "<short description>",
        "chart_type": "histogram | scatter | bar | line | box | null",
        "x": "<column or null>",
        "y": "<column or null>",
        "color": "<column or null>",
        "facet_col": "<column or null>",
        "facet_row": "<column or null>",
        "aggregations": ["<optional aggregations>"] ,
                "parameters": {
                    "nbins": 40,
                    "derived_columns": ["release_month"],
                    "result_columns": ["avg_popularity"]
                },
        "code": "def transform(df): ... return {'figure': fig, 'preview': [...]} ",
        "warnings": ["<optional warnings>"] ,
        "explanation": "<reasoning>"
    }
    return json.dumps(schema, indent=2)


def _build_chart_prompt(query: str, profile: Dict[str, Any]) -> str:
    schema = _build_chart_schema_json()
    profile_json = json.dumps(profile, indent=2)
    example = textwrap.dedent(
        """
        Example output for request "scatter plot of popularity vs revenue colored by genre":
        {
          "intent_summary": "Scatter plot of popularity against revenue grouped by genre",
          "chart_type": "scatter",
          "x": "popularity",
          "y": "revenue",
          "color": "genre",
          "aggregations": [],
          "parameters": {"trendline": "ols"},
          "code": "def transform(df):\n    df = df.copy()\n    fig = px.scatter(df, x='popularity', y='revenue', color='genre', trendline='ols')\n    preview = df[['popularity','revenue','genre']].head(5).to_dict(orient='records')\n    return {\n        'figure': fig,\n        'preview': preview,\n        'summary': 'Scatter plot showing popularity vs revenue by genre.'\n    }",
          "warnings": [],
          "explanation": "Using scatter plot to show relationship between popularity and revenue colored by genre."
        }
        """
    ).strip()

    instructions = textwrap.dedent(
        f"""
        You are a data visualization expert. Translate the user's request into executable Python code that builds a Plotly Express chart.
        Always define a function transform(df) that returns at least a 'figure' (Plotly figure), a 'preview' (list of first rows involved), and a 'summary'.
        Only use pandas, numpy, and plotly.express/plotly.graph_objects. Do not import other libraries.
        The runtime already provides pandas as pd, numpy as np, plotly.express as px, and plotly.graph_objects as go. Do NOT include any import statements in the generated code.
        Select columns carefully from the dataset profile. If the user names columns, map them exactly. If the request is ambiguous, make a reasonable assumption and explain it in 'explanation'.
        If the chart requires aggregation (e.g., average, sum), compute it inside transform(df) before plotting and list the operations in 'aggregations'.
        Make sure the code handles missing values gracefully (dropna on required columns if appropriate) and avoids modifying the original df in place.
        When you derive new helper columns (e.g., month extracted from release_date), create them inside transform(df) and list them under parameters. You must also return the names of any newly created columns in parameters.derived_columns so validation accepts them.
        When you calculate aggregated outputs that introduce new column names (e.g., avg_popularity), include those names in parameters.result_columns so downstream validation allows them.
        If the request cannot be fulfilled with available columns, return code that simply produces an empty plot and explain why in 'explanation'.
        You MUST output JSON matching the schema, without extra commentary or backticks.

        Dataset profile:
        {profile_json}

        User request:
        {query}

        Schema:
        {schema}

        {example}
        """
    ).strip()
    return instructions


def generate_chart_with_gemini(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    profile = profile_dataframe(df)
    prompt = _build_chart_prompt(query=query, profile=profile)
    try:
        raw_text = _invoke_gemini(prompt)
    except PlanValidationError as exc:
        error_message = str(exc)
        if "MAX_TOKENS" not in error_message:
            raise

        logger.warning("Gemini hit max tokens for chart prompt. Retrying with compact profile.")

        compact_profile = {
            "columns": profile.get("columns", []),
            "dtypes": profile.get("dtypes", {}),
            "missing_percent": profile.get("missing_percent", {}),
            "n_unique": profile.get("n_unique", {}),
            "sample_rows": profile.get("sample_rows", [])[:2],
        }

        compact_prompt = _build_chart_prompt(query=query, profile=compact_profile)
        raw_text = _invoke_gemini(compact_prompt, max_output_tokens=6144)

    payload_dict = _extract_json_payload(raw_text)

    try:
        chart_plan = ChartPlanPayload.model_validate(payload_dict)
    except ValidationError as exc:
        raise PlanValidationError(f"Chart generation schema validation failed: {exc}") from exc

    chart_plan.validate_against_dataframe(df)

    plan_dict = chart_plan.model_dump(exclude_none=True)
    plan_id = str(uuid.uuid4())

    return {
        "plan": plan_dict,
        "plan_id": plan_id,
        "profile": profile,
        "raw_response": raw_text,
    }


def get_cleaning_suggestions(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Retain simple heuristic suggestions as fallback."""
    steps: List[Dict[str, Any]] = []

    missing = profile.get("missing_percent", {})
    dtypes = profile.get("dtypes", {})

    for col, miss in missing.items():
        if miss and miss > 0:
            dtype = dtypes.get(col, "")
            method = "median" if any(token in dtype for token in ("float", "int")) else "mode"
            steps.append(
                {
                    "description": f"Fill missing values in '{col}' using {method}",
                    "column": col,
                    "action": "fill_missing",
                    "parameters": {"method": method},
                }
            )

    steps.append(
        {
            "description": "Drop duplicate rows",
            "column": None,
            "action": "drop_duplicates",
            "parameters": {},
        }
    )

    return {"steps": steps}


def get_chart_spec(query: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Fallback NL -> chart spec parser when LLM is unavailable."""
    q = query.lower()

    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if not is_numeric_dtype(df[c])]

    chart_type = "histogram"
    x = None
    y = None
    color = None

    if "scatter" in q:
        chart_type = "scatter"
    elif "bar" in q:
        chart_type = "bar"
    elif "line" in q:
        chart_type = "line"
    elif "hist" in q or "distribution" in q:
        chart_type = "histogram"

    mentioned_cols = [col for col in df.columns if col.lower() in q]

    if chart_type == "histogram":
        x = mentioned_cols[0] if mentioned_cols else (numeric_cols[0] if numeric_cols else df.columns[0])

    elif chart_type in {"scatter", "line"}:
        if len(mentioned_cols) >= 2:
            x, y = mentioned_cols[0], mentioned_cols[1]
        elif len(numeric_cols) >= 2:
            x, y = numeric_cols[0], numeric_cols[1]
        elif len(df.columns) >= 2:
            x, y = df.columns[0], df.columns[1]

    elif chart_type == "bar":
        if mentioned_cols:
            x = mentioned_cols[0]
            y = mentioned_cols[1] if len(mentioned_cols) > 1 else None
        else:
            cat = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
            num = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns) > 1 else None)
            x, y = cat, num

    if " by " in q:
        match = re.search(r"by\s+([a-zA-Z0-9_]+)", q)
        if match:
            possible = match.group(1)
            for col in df.columns:
                if col.lower() == possible.lower():
                    color = col
                    break

    return {
        "chart_type": chart_type,
        "x": x,
        "y": y,
        "color": color,
    }


__all__ = [
    "ChartPlanPayload",
    "CodePlanPayload",
    "PlanPayload",
    "PlanValidationError",
    "generate_chart_with_gemini",
    "generate_code_with_gemini",
    "generate_plan_with_gemini",
    "get_cleaning_suggestions",
    "get_chart_spec",
    "profile_dataframe",
]
