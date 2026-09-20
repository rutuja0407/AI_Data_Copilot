import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


ALLOWED_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "round": round,
    "enumerate": enumerate,
    "range": range,
    "sorted": sorted,
    "zip": zip,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "any": any,
    "all": all,
}


class CodeExecutionError(Exception):
    """Raised when generated code cannot be executed safely."""


def _build_execution_globals() -> Dict[str, Any]:
    safe_globals: Dict[str, Any] = {
        "__builtins__": ALLOWED_BUILTINS,
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
    }
    return safe_globals


def execute_generated_code(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Execute generated Python code in a constrained namespace.

    The code must define a function named `transform(df)` that returns a dict.
    The returned dict may include:
      - "dataframe" or "df": resulting pandas DataFrame
      - "preview": serialisable preview data
      - "summary": textual summary
      - "result": arbitrary serialisable payload
    """

    if "transform" not in code:
        raise CodeExecutionError("Generated code must define a function named 'transform'.")

    compiled = compile(code, filename="<generated>", mode="exec")
    exec_globals = _build_execution_globals()
    exec_locals: Dict[str, Any] = {}

    try:
        exec(compiled, exec_globals, exec_locals)
    except Exception as exc:  # pragma: no cover - depends on generated code
        raise CodeExecutionError(f"Failed to compile generated code: {exc}") from exc

    transform = exec_locals.get("transform") or exec_globals.get("transform")
    if transform is None or not callable(transform):
        raise CodeExecutionError("Generated code must define callable transform(df).")

    try:
        result = transform(df.copy())
    except Exception as exc:  # pragma: no cover - depends on generated code
        raise CodeExecutionError(f"Generated transform execution failed: {exc}") from exc

    if result is None:
        raise CodeExecutionError("transform(df) must return a dict with at least a dataframe or preview.")

    if not isinstance(result, dict):
        raise CodeExecutionError("transform(df) must return a dict.")

    output: Dict[str, Any] = {}

    dataframe: Optional[pd.DataFrame] = None
    if "dataframe" in result and isinstance(result["dataframe"], pd.DataFrame):
        dataframe = result["dataframe"]
    elif "df" in result and isinstance(result["df"], pd.DataFrame):
        dataframe = result["df"]

    if dataframe is not None:
        output["dataframe"] = dataframe

    if "preview" in result:
        output["preview"] = result["preview"]

    if "summary" in result:
        output["summary"] = result["summary"]

    if "result" in result:
        output["result"] = result["result"]

    if "messages" in result:
        output["messages"] = result["messages"]

    figure_obj = result.get("figure")
    if figure_obj is not None:
        if isinstance(figure_obj, dict):
            output["figure"] = figure_obj
        elif hasattr(figure_obj, "to_json"):
            try:
                output["figure"] = json.loads(figure_obj.to_json())
            except (TypeError, ValueError):
                try:
                    output["figure"] = figure_obj.to_plotly_json()  # type: ignore[attr-defined]
                except AttributeError as exc:
                    raise CodeExecutionError("Returned figure could not be serialised to JSON.") from exc
        elif hasattr(figure_obj, "to_plotly_json"):
            output["figure"] = figure_obj.to_plotly_json()
        elif hasattr(figure_obj, "to_dict"):
            output["figure"] = figure_obj.to_dict()
        else:
            raise CodeExecutionError("Returned figure is not a Plotly figure or dict.")

    if "figure_json" in result and "figure" not in output:
        output["figure"] = result["figure_json"]

    if not output:
        raise CodeExecutionError("transform(df) returned an empty payload. Expected dataframe or summary.")

    return output
