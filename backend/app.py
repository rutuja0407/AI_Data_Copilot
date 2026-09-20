import io
import logging
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, ValidationError

from .storage import (
    DATAFRAMES,
    append_audit_entry,
    get_plan,
    register_plan,
    snapshot_dataframe,
)
from .services.ai import (
    ChartPlanPayload,
    CodePlanPayload,
    PlanPayload,
    PlanValidationError,
    generate_chart_with_gemini,
    generate_code_with_gemini,
    generate_plan_with_gemini,
    get_cleaning_suggestions,
)
from .services.cleaner import apply_steps
from .services.executor import CodeExecutionError, execute_generated_code


app = FastAPI(title="AI Data Cleaning & Visualization Copilot")
logger = logging.getLogger(__name__)

# CORS so Streamlit (different port) can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LLMPlanRequest(BaseModel):
    query: str


class ExecutePlanRequest(BaseModel):
    plan_id: str
    dry_run: bool = False


class LLMCodeRequest(BaseModel):
    query: str


class ExecuteCodeRequest(BaseModel):
    plan_id: str
    dry_run: bool = False


class LLMChartRequest(BaseModel):
    query: str


class ExecuteChartRequest(BaseModel):
    plan_id: str


def summarize_changes(before: pd.DataFrame, after: pd.DataFrame, steps: List[dict]) -> dict:
    impacted_columns = sorted({step.get("column") for step in steps if step.get("column")})

    summary = {
        "rows_before": int(before.shape[0]),
        "rows_after": int(after.shape[0]),
        "row_delta": int(after.shape[0] - before.shape[0]),
        "columns_added": sorted(set(after.columns) - set(before.columns)),
        "columns_removed": sorted(set(before.columns) - set(after.columns)),
        "columns_impacted": impacted_columns,
        "missing_percent_delta": {},
    }

    for column in impacted_columns:
        if column in before.columns and column in after.columns:
            summary["missing_percent_delta"][column] = {
                "before": float(before[column].isna().mean()),
                "after": float(after[column].isna().mean()),
            }

    return summary


def dataframe_preview(df: pd.DataFrame, limit: int = 5) -> List[dict]:
    limited = df.head(limit) if limit is not None else df
    sanitized = limited.replace([np.inf, -np.inf], pd.NA)
    sanitized = sanitized.where(pd.notna(sanitized), None)
    return sanitized.to_dict(orient="records")


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read CSV file")

    file_id = file.filename  # simple id; can be replaced with UUID
    DATAFRAMES[file_id] = df
    snapshot_dataframe(file_id)

    info = {
        "file_id": file_id,
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }
    return info


@app.get("/profile/{file_id}")
async def profile_data(file_id: str):
    if file_id not in DATAFRAMES:
        raise HTTPException(status_code=404, detail="File not found")

    df = DATAFRAMES[file_id]

    # Basic describe stats
    desc = df.describe(include="all").transpose().fillna("").to_dict(orient="index")

    meta = {
        "missing_percent": df.isna().mean().round(4).to_dict(),
        "n_unique": df.nunique().to_dict(),
    }

    return {"describe": desc, "meta": meta}


@app.get("/suggest_cleaning/{file_id}")
async def suggest_cleaning(file_id: str):
    if file_id not in DATAFRAMES:
        raise HTTPException(status_code=404, detail="File not found")

    df = DATAFRAMES[file_id]
    profile = {
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_percent": df.isna().mean().round(4).to_dict(),
        "n_unique": df.nunique().to_dict(),
    }

    steps = get_cleaning_suggestions(profile)
    return steps


@app.post("/llm/plan/{file_id}")
async def llm_generate_plan(file_id: str, payload: LLMPlanRequest):
    if file_id not in DATAFRAMES:
        raise HTTPException(status_code=404, detail="File not found")

    df = DATAFRAMES[file_id]

    try:
        llm_output = generate_plan_with_gemini(query=payload.query, df=df)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PlanValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    plan_id = llm_output["plan_id"]
    register_plan(
        plan_id,
        {
            "plan_id": plan_id,
            "file_id": file_id,
            "query": payload.query,
            "plan": llm_output["plan"],
            "profile": llm_output["profile"],
            "raw_response": llm_output["raw_response"],
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    )

    append_audit_entry(
        {
            "event": "plan_generated",
            "plan_id": plan_id,
            "file_id": file_id,
            "query": payload.query,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )

    logger.info("LLM plan %s generated for file %s", plan_id, file_id)

    return {
        "plan_id": plan_id,
        "plan": llm_output["plan"],
        "profile": llm_output["profile"],
    }


@app.post("/llm/code/{file_id}")
async def llm_generate_code(file_id: str, payload: LLMCodeRequest):
    if file_id not in DATAFRAMES:
        raise HTTPException(status_code=404, detail="File not found")

    df = DATAFRAMES[file_id]

    try:
        code_output = generate_code_with_gemini(query=payload.query, df=df)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PlanValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    plan_id = code_output["plan_id"]
    register_plan(
        plan_id,
        {
            "plan_id": plan_id,
            "plan_type": "code",
            "file_id": file_id,
            "query": payload.query,
            "plan": code_output["plan"],
            "profile": code_output["profile"],
            "raw_response": code_output["raw_response"],
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    )

    append_audit_entry(
        {
            "event": "code_plan_generated",
            "plan_id": plan_id,
            "file_id": file_id,
            "query": payload.query,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )

    logger.info("LLM code plan %s generated for file %s", plan_id, file_id)

    return jsonable_encoder({
        "plan_id": plan_id,
        "plan": code_output["plan"],
        "profile": code_output["profile"],
    })


@app.post("/llm/chart/{file_id}")
async def llm_generate_chart(file_id: str, payload: LLMChartRequest):
    if file_id not in DATAFRAMES:
        raise HTTPException(status_code=404, detail="File not found")

    df = DATAFRAMES[file_id]

    try:
        chart_output = generate_chart_with_gemini(query=payload.query, df=df)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PlanValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    plan_id = chart_output["plan_id"]
    register_plan(
        plan_id,
        {
            "plan_id": plan_id,
            "plan_type": "chart",
            "file_id": file_id,
            "query": payload.query,
            "plan": chart_output["plan"],
            "profile": chart_output["profile"],
            "raw_response": chart_output["raw_response"],
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    )

    append_audit_entry(
        {
            "event": "chart_plan_generated",
            "plan_id": plan_id,
            "file_id": file_id,
            "query": payload.query,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )

    logger.info("LLM chart plan %s generated for file %s", plan_id, file_id)

    return jsonable_encoder({
        "plan_id": plan_id,
        "plan": chart_output["plan"],
        "profile": chart_output["profile"],
    })


@app.post("/apply_cleaning/{file_id}")
async def apply_cleaning(file_id: str, steps: dict = Body(...)):
    if file_id not in DATAFRAMES:
        raise HTTPException(status_code=404, detail="File not found")

    df = DATAFRAMES[file_id]
    snapshot_dataframe(file_id)
    df_clean = apply_steps(df, steps)
    DATAFRAMES[file_id] = df_clean

    return {
        "rows": df_clean.shape[0],
        "cols": df_clean.shape[1]
    }


@app.post("/llm/execute/{file_id}")
async def llm_execute_plan(file_id: str, payload: ExecutePlanRequest):
    if file_id not in DATAFRAMES:
        raise HTTPException(status_code=404, detail="File not found")

    plan_entry = get_plan(payload.plan_id)
    if plan_entry is None or plan_entry.get("file_id") != file_id:
        raise HTTPException(status_code=404, detail="Plan not found for this dataset")

    df_current = DATAFRAMES[file_id]

    try:
        plan_model = PlanPayload.model_validate(plan_entry["plan"])
        plan_model.validate_against_dataframe(df_current)
    except (ValidationError, PlanValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Plan validation failed: {exc}") from exc

    plan_steps: List[dict] = plan_entry["plan"].get("steps", [])
    before = df_current.copy(deep=True)
    after = apply_steps(before.copy(deep=True), {"steps": plan_steps})

    summary = summarize_changes(before, after, plan_steps)

    append_audit_entry(
        {
            "event": "plan_preview" if payload.dry_run else "plan_applied",
            "plan_id": payload.plan_id,
            "file_id": file_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dry_run": payload.dry_run,
            "summary": summary,
        }
    )

    if payload.dry_run:
        return jsonable_encoder({
            "applied": False,
            "summary": summary,
            "preview": dataframe_preview(after),
        })

    snapshot_dataframe(file_id)
    DATAFRAMES[file_id] = after
    plan_entry["applied_at"] = datetime.utcnow().isoformat() + "Z"
    plan_entry["summary"] = summary
    plan_entry["applied"] = True

    logger.info("LLM plan %s applied to file %s", payload.plan_id, file_id)

    return jsonable_encoder({
        "applied": True,
        "summary": summary,
        "rows": after.shape[0],
        "cols": after.shape[1],
    })


@app.post("/llm/code/execute/{file_id}")
async def llm_execute_code(file_id: str, payload: ExecuteCodeRequest):
    if file_id not in DATAFRAMES:
        raise HTTPException(status_code=404, detail="File not found")

    plan_entry = get_plan(payload.plan_id)
    if plan_entry is None or plan_entry.get("file_id") != file_id:
        raise HTTPException(status_code=404, detail="Plan not found for this dataset")

    if plan_entry.get("plan_type") != "code":
        raise HTTPException(status_code=400, detail="Plan is not a code execution plan")

    df_current = DATAFRAMES[file_id]

    try:
        code_model = CodePlanPayload.model_validate(plan_entry["plan"]).validate_against_dataframe(df_current)
    except (ValidationError, PlanValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Code plan validation failed: {exc}") from exc

    before = df_current.copy(deep=True)

    try:
        execution_output = execute_generated_code(code_model.code, before.copy(deep=True))
    except CodeExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    after = execution_output.get("dataframe")
    if after is None:
        raise HTTPException(status_code=400, detail="Generated code did not return a dataframe result.")

    summary = summarize_changes(
        before,
        after,
        [{"column": col} for col in code_model.columns_used],
    )

    response_payload = {
        "summary": summary,
        "preview": execution_output.get("preview")
        or dataframe_preview(after),
        "result": execution_output.get("result"),
        "messages": execution_output.get("messages"),
        "explanation": code_model.explanation,
        "warnings": code_model.warnings,
    }

    append_audit_entry(
        {
            "event": "code_plan_preview" if payload.dry_run else "code_plan_applied",
            "plan_id": payload.plan_id,
            "file_id": file_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dry_run": payload.dry_run,
            "summary": summary,
        }
    )

    if payload.dry_run:
        response_payload["applied"] = False
        if isinstance(response_payload.get("preview"), pd.DataFrame):
            response_payload["preview"] = dataframe_preview(response_payload["preview"])  # type: ignore[index]
        return jsonable_encoder(response_payload)

    snapshot_dataframe(file_id)
    DATAFRAMES[file_id] = after
    plan_entry["applied_at"] = datetime.utcnow().isoformat() + "Z"
    plan_entry["summary"] = summary
    plan_entry["applied"] = True

    logger.info("LLM code plan %s applied to file %s", payload.plan_id, file_id)

    if isinstance(response_payload.get("preview"), pd.DataFrame):
        response_payload["preview"] = dataframe_preview(response_payload["preview"])  # type: ignore[index]
    response_payload["applied"] = True
    response_payload["rows"] = after.shape[0]
    response_payload["cols"] = after.shape[1]
    return jsonable_encoder(response_payload)


@app.get("/download_df/{file_id}")
async def download_df(file_id: str):
    if file_id not in DATAFRAMES:
        raise HTTPException(status_code=404, detail="File not found")

    df = DATAFRAMES[file_id]
    csv_text = df.to_csv(index=False)
    return Response(content=csv_text, media_type="text/csv")


@app.post("/llm/chart/execute/{file_id}")
async def llm_execute_chart(file_id: str, payload: ExecuteChartRequest):
    if file_id not in DATAFRAMES:
        raise HTTPException(status_code=404, detail="File not found")

    plan_entry = get_plan(payload.plan_id)
    if plan_entry is None or plan_entry.get("file_id") != file_id:
        raise HTTPException(status_code=404, detail="Chart plan not found for this dataset")

    if plan_entry.get("plan_type") != "chart":
        raise HTTPException(status_code=400, detail="Plan is not a chart generation plan")

    df = DATAFRAMES[file_id]

    try:
        chart_model = ChartPlanPayload.model_validate(plan_entry["plan"]).validate_against_dataframe(df)
    except (ValidationError, PlanValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Chart plan validation failed: {exc}") from exc

    try:
        execution_output = execute_generated_code(chart_model.code, df.copy(deep=True))
    except CodeExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    figure = execution_output.get("figure")
    if figure is None:
        raise HTTPException(status_code=400, detail="Generated chart code did not return a figure.")

    preview = execution_output.get("preview")
    if isinstance(preview, pd.DataFrame):
        preview_payload = dataframe_preview(preview)
    else:
        preview_payload = preview

    response_payload = {
        "figure": figure,
        "preview": preview_payload,
        "summary": execution_output.get("summary"),
        "result": execution_output.get("result"),
        "messages": execution_output.get("messages"),
        "warnings": chart_model.warnings,
        "explanation": chart_model.explanation,
        "intent_summary": chart_model.intent_summary,
        "chart_type": chart_model.chart_type,
        "parameters": chart_model.parameters,
        "aggregations": chart_model.aggregations,
    }

    append_audit_entry(
        {
            "event": "chart_plan_executed",
            "plan_id": payload.plan_id,
            "file_id": file_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )

    logger.info("LLM chart plan %s executed for file %s", payload.plan_id, file_id)

    return jsonable_encoder(response_payload)
