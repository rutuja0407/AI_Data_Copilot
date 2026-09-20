import streamlit as st
import requests
import pandas as pd
from io import StringIO
from typing import Optional
import plotly.graph_objects as go

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Data Copilot", layout="wide")
st.title("🧹📊 AI Data Cleaning & Visualization Copilot")

st.write("Upload a CSV file, get cleaning suggestions, and create visualizations via natural language.")


if "file_id" not in st.session_state:
    st.session_state["file_id"] = None

if "llm_plan" not in st.session_state:
    st.session_state["llm_plan"] = None

if "chart_plan" not in st.session_state:
    st.session_state["chart_plan"] = None

if "chart_result" not in st.session_state:
    st.session_state["chart_result"] = None

if "chart_query" not in st.session_state:
    st.session_state["chart_query"] = ""

if "llm_dry_run" not in st.session_state:
    st.session_state["llm_dry_run"] = None

if "llm_query" not in st.session_state:
    st.session_state["llm_query"] = ""

if "code_plan" not in st.session_state:
    st.session_state["code_plan"] = None

if "code_preview" not in st.session_state:
    st.session_state["code_preview"] = None

if "code_query" not in st.session_state:
    st.session_state["code_query"] = ""


uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
    with st.spinner("Uploading and reading file..."):
        resp = requests.post(f"{BACKEND_URL}/upload_csv", files=files)

    if resp.status_code == 200:
        info = resp.json()
        st.session_state["file_id"] = info["file_id"]
        st.success(f"File uploaded: {info['file_id']}")
        st.write(f"Rows: {info['rows']} | Columns: {info['cols']}")
        st.write("Columns and types:", info["dtypes"])
    else:
        st.error(f"Error: {resp.text}")


file_id = st.session_state.get("file_id", None)

if file_id:
    st.markdown("---")
    cols = st.columns(2)

    # LEFT: Profiling + Cleaning
    with cols[0]:
        st.subheader("🔍 Data Profiling & Cleaning")

        if st.button("Generate Profile"):
            with st.spinner("Generating profile..."):
                prof_resp = requests.get(f"{BACKEND_URL}/profile/{file_id}")
            if prof_resp.status_code == 200:
                prof = prof_resp.json()
                st.write("**Missing % per column**")
                st.json(prof["meta"]["missing_percent"])
                st.write("**Unique values per column**")
                st.json(prof["meta"]["n_unique"])
            else:
                st.error("Profile error: " + prof_resp.text)

        if st.button("Get AI Cleaning Suggestions"):
            with st.spinner("Getting suggestions..."):
                sugg_resp = requests.get(f"{BACKEND_URL}/suggest_cleaning/{file_id}")

            if sugg_resp.status_code == 200:
                suggestions = sugg_resp.json()
                st.write("**Suggested Cleaning Steps**")
                st.json(suggestions)

                if st.button("Apply Suggested Cleaning"):
                    with st.spinner("Applying cleaning steps..."):
                        apply_resp = requests.post(
                            f"{BACKEND_URL}/apply_cleaning/{file_id}",
                            json=suggestions,
                        )
                    if apply_resp.status_code == 200:
                        info = apply_resp.json()
                        st.success(
                            f"Cleaning applied! New shape: {info['rows']} rows x {info['cols']} cols"
                        )
                    else:
                        st.error("Error applying cleaning: " + apply_resp.text)
            else:
                st.error("Error getting suggestions: " + sugg_resp.text)

        st.markdown("---")
        st.subheader("🤖 Gemini Cleaning Plan")

        llm_query = st.text_area(
            "Describe how you want to transform the dataset",
            key="llm_query",
            placeholder="e.g. Fill missing ages with median, clip salary outliers, drop rows missing country",
            height=120,
        )

        plan_col1, plan_col2 = st.columns([1, 1])
        with plan_col1:
            if st.button("Generate Plan", key="llm_generate_plan"):
                if not llm_query.strip():
                    st.warning("Please describe the cleaning or analysis task.")
                else:
                    with st.spinner("Requesting Gemini plan..."):
                        resp = requests.post(
                            f"{BACKEND_URL}/llm/plan/{file_id}",
                            json={"query": llm_query},
                        )
                    if resp.status_code == 200:
                        st.session_state["llm_plan"] = resp.json()
                        st.session_state["llm_dry_run"] = None
                        st.success("Plan generated! Review below before applying.")
                    else:
                        st.error(f"LLM error: {resp.text}")
        with plan_col2:
            if st.button("Discard Plan", key="llm_discard_plan"):
                if st.session_state.get("llm_plan") is None:
                    st.info("No plan to discard.")
                else:
                    st.session_state["llm_plan"] = None
                    st.session_state["llm_dry_run"] = None
                    st.success("Cleared current plan preview.")

        plan_state = st.session_state.get("llm_plan")
        if plan_state:
            plan = plan_state.get("plan", {})
            plan_id = plan_state.get("plan_id")
            st.markdown(f"**Plan ID:** `{plan_id}`")
            overview = plan.get("plan_overview")
            if overview:
                st.info(overview)
            confidence = plan.get("confidence")
            if confidence is not None:
                st.metric("LLM Confidence", f"{confidence:.2f}")

            steps = plan.get("steps", [])
            if steps:
                st.write("**Proposed Steps**")
                steps_df = pd.DataFrame(steps)
                st.dataframe(steps_df, use_container_width=True)
                action_cols = st.columns([1, 1, 1])
                with action_cols[0]:
                    if st.button("Preview Dry Run", key="llm_dry_run_button"):
                        with st.spinner("Simulating plan without applying..."):
                            exec_resp = requests.post(
                                f"{BACKEND_URL}/llm/execute/{file_id}",
                                json={"plan_id": plan_id, "dry_run": True},
                            )
                        if exec_resp.status_code == 200:
                            st.session_state["llm_dry_run"] = exec_resp.json()
                            st.success("Dry-run completed. See summary below.")
                        else:
                            st.error(f"Dry-run failed: {exec_resp.text}")
                with action_cols[1]:
                    if st.button("Apply Plan", key="llm_apply_button"):
                        with st.spinner("Applying plan to dataset..."):
                            exec_resp = requests.post(
                                f"{BACKEND_URL}/llm/execute/{file_id}",
                                json={"plan_id": plan_id, "dry_run": False},
                            )
                        if exec_resp.status_code == 200:
                            result = exec_resp.json()
                            st.success("Plan applied successfully!")
                            st.json(result.get("summary", {}))
                            st.session_state["llm_plan"] = None
                            st.session_state["llm_dry_run"] = None
                        else:
                            st.error(f"Plan application failed: {exec_resp.text}")
                with action_cols[2]:
                    if st.button("Clear Dry-Run", key="llm_clear_dry_run"):
                        st.session_state["llm_dry_run"] = None
            else:
                st.warning("No executable steps returned. Confidence is likely low.")

            chart = plan.get("chart")
            if chart:
                st.write("**Suggested Chart**")
                st.json(chart)

            dry_run_state = st.session_state.get("llm_dry_run")
            if dry_run_state:
                st.write("**Dry-Run Summary**")
                st.json(dry_run_state.get("summary", {}))
                preview_records = dry_run_state.get("preview", [])
                if preview_records:
                    st.write("**Sample Preview After Plan (first 5 rows)**")
                    st.dataframe(pd.DataFrame(preview_records))

        st.markdown("---")
        st.subheader("🧠 Gemini Code Agent")

        code_query = st.text_area(
            "Ask anything about the dataset (code generation)",
            key="code_query",
            placeholder="e.g. 'Compute average rating by genre' or 'Create a pivot of revenue by country and year'",
            height=120,
        )

        code_buttons = st.columns([1, 1])
        with code_buttons[0]:
            if st.button("Generate Code", key="code_generate"):
                if not code_query.strip():
                    st.warning("Please describe the transformation or analysis you need.")
                else:
                    with st.spinner("Requesting Gemini code plan..."):
                        resp = requests.post(
                            f"{BACKEND_URL}/llm/code/{file_id}",
                            json={"query": code_query},
                        )
                    if resp.status_code == 200:
                        st.session_state["code_plan"] = resp.json()
                        st.session_state["code_preview"] = None
                        st.success("Code plan generated! Review below before running.")
                    else:
                        st.error(f"LLM error: {resp.text}")
        with code_buttons[1]:
            if st.button("Discard Code Plan", key="code_discard"):
                if st.session_state.get("code_plan") is None:
                    st.info("No code plan to discard.")
                else:
                    st.session_state["code_plan"] = None
                    st.session_state["code_preview"] = None
                    st.success("Cleared code plan preview.")

        code_state = st.session_state.get("code_plan")
        if code_state:
            code_plan = code_state.get("plan", {})
            code_plan_id = code_state.get("plan_id")
            st.markdown(f"**Code Plan ID:** `{code_plan_id}`")

            if code_plan.get("intent_summary"):
                st.info(code_plan["intent_summary"])

            if code_plan.get("explanation"):
                st.write("**Explanation**")
                st.write(code_plan["explanation"])

            if code_plan.get("columns_used"):
                st.write("**Columns Used**")
                st.write(code_plan["columns_used"])

            if code_plan.get("actions"):
                st.write("**Detected Actions**")
                st.write(code_plan["actions"])

            if code_plan.get("warnings"):
                st.warning("\n".join(code_plan["warnings"]))

            st.write("**Generated Code**")
            st.code(code_plan.get("code", ""), language="python")

            code_action_cols = st.columns([1, 1, 1])
            with code_action_cols[0]:
                if st.button("Dry Run Code", key="code_dry_run"):
                    with st.spinner("Running code in dry-run mode..."):
                        exec_resp = requests.post(
                            f"{BACKEND_URL}/llm/code/execute/{file_id}",
                            json={"plan_id": code_plan_id, "dry_run": True},
                        )
                    if exec_resp.status_code == 200:
                        st.session_state["code_preview"] = exec_resp.json()
                        st.success("Dry-run completed. Review the summary below.")
                    else:
                        st.error(f"Dry-run failed: {exec_resp.text}")
            with code_action_cols[1]:
                if st.button("Execute Code", key="code_execute"):
                    with st.spinner("Executing generated code..."):
                        exec_resp = requests.post(
                            f"{BACKEND_URL}/llm/code/execute/{file_id}",
                            json={"plan_id": code_plan_id, "dry_run": False},
                        )
                    if exec_resp.status_code == 200:
                        result = exec_resp.json()
                        st.success("Code executed and dataset updated.")
                        st.json(result.get("summary", {}))
                        if result.get("preview"):
                            st.write("**Result Preview (first rows)**")
                            st.dataframe(pd.DataFrame(result["preview"]))
                        st.session_state["code_plan"] = None
                        st.session_state["code_preview"] = None
                    else:
                        st.error(f"Code execution failed: {exec_resp.text}")
            with code_action_cols[2]:
                if st.button("Clear Dry-Run Result", key="code_clear_dry_run"):
                    st.session_state["code_preview"] = None

            code_preview_state = st.session_state.get("code_preview")
            if code_preview_state:
                st.write("**Dry-Run Summary**")
                st.json(code_preview_state.get("summary", {}))
                if code_preview_state.get("preview"):
                    st.write("**Dry-Run Preview (first rows)**")
                    st.dataframe(pd.DataFrame(code_preview_state["preview"]))
                if code_preview_state.get("result"):
                    st.write("**Result Payload**")
                    st.json(code_preview_state["result"])
                if code_preview_state.get("messages"):
                    st.write("**Messages**")
                    st.write(code_preview_state["messages"])

        if st.button("Preview Current Data (first 10 rows)"):
            df_resp = requests.get(f"{BACKEND_URL}/download_df/{file_id}")
            if df_resp.status_code == 200:
                df = pd.read_csv(StringIO(df_resp.text))
                st.dataframe(df.head(10))
            else:
                st.error("Error downloading data: " + df_resp.text)

    # RIGHT: Natural Language Visualization
    with cols[1]:
        st.subheader("📈 Natural Language to Visualization")
        chart_query = st.text_input(
            "Ask for a chart",
            key="chart_query",
            placeholder="e.g. 'Box plot of vote_average by language' or 'Line of popularity over time'",
        )

        chart_buttons = st.columns([1, 1])
        with chart_buttons[0]:
            if st.button("Generate Chart", key="chart_generate"):
                if not chart_query.strip():
                    st.warning("Please type a query.")
                else:
                    with st.spinner("Requesting Gemini chart plan..."):
                        spec_resp = requests.post(
                            f"{BACKEND_URL}/llm/chart/{file_id}",
                            json={"query": chart_query},
                        )

                    if spec_resp.status_code != 200:
                        st.error("Error from backend: " + spec_resp.text)
                    else:
                        st.session_state["chart_plan"] = spec_resp.json()
                        st.session_state["chart_result"] = None
                        st.success("Chart plan generated. Review below.")

        with chart_buttons[1]:
            if st.button("Clear Chart Plan", key="chart_clear"):
                st.session_state["chart_plan"] = None
                st.session_state["chart_result"] = None
                st.success("Cleared chart state.")

        chart_state = st.session_state.get("chart_plan")
        plan_explanation: Optional[str] = None
        if chart_state:
            chart_plan = chart_state.get("plan", {})
            chart_plan_id = chart_state.get("plan_id")
            st.markdown(f"**Chart Plan ID:** `{chart_plan_id}`")

            if chart_plan.get("intent_summary"):
                st.info(chart_plan["intent_summary"])

            plan_explanation = chart_plan.get("explanation")

            spec_cols = st.columns([1, 1])
            with spec_cols[0]:
                st.write("**Chart Type**")
                st.write(chart_plan.get("chart_type"))
            with spec_cols[1]:
                st.write("**Core Columns**")
                st.write(
                    {
                        "x": chart_plan.get("x"),
                        "y": chart_plan.get("y"),
                        "color": chart_plan.get("color"),
                        "facet_col": chart_plan.get("facet_col"),
                        "facet_row": chart_plan.get("facet_row"),
                    }
                )

            if chart_plan.get("aggregations"):
                st.write("**Aggregations**")
                st.write(chart_plan["aggregations"])

            if chart_plan.get("parameters"):
                st.write("**Parameters**")
                st.json(chart_plan["parameters"])

            if chart_plan.get("warnings"):
                st.warning("\n".join(chart_plan["warnings"]))

            if chart_plan.get("explanation"):
                st.write("**Explanation**")
                st.write(chart_plan["explanation"])

            st.write("**Generated Code**")
            st.code(chart_plan.get("code", ""), language="python")

            if st.button("Render Chart", key="chart_execute"):
                with st.spinner("Executing chart code..."):
                    exec_resp = requests.post(
                        f"{BACKEND_URL}/llm/chart/execute/{file_id}",
                        json={"plan_id": chart_plan_id},
                    )
                if exec_resp.status_code != 200:
                    st.error("Chart execution failed: " + exec_resp.text)
                else:
                    st.session_state["chart_result"] = exec_resp.json()
                    st.success("Chart rendered successfully.")

        chart_result = st.session_state.get("chart_result")
        if chart_result:
            figure_payload = chart_result.get("figure")
            if figure_payload:
                try:
                    fig = go.Figure(figure_payload)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as exc:
                    st.error(f"Unable to render figure: {exc}")

            explanation_text = chart_result.get("explanation") or plan_explanation
            if explanation_text:
                st.write("**Explanation**")
                st.write(explanation_text)

            if chart_result.get("summary"):
                st.write("**Chart Summary**")
                st.write(chart_result["summary"])
