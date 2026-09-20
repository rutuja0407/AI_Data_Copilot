# AI Data Cleaning & Visualization Copilot

An interactive data assistant for uploading CSV files, profiling datasets, applying cleaning steps, and generating cleaning, analytics, and visualization plans with Google Gemini.

## Features

- Upload CSV files from the Streamlit interface
- Inspect missing values, unique values, data types, and descriptive statistics
- Generate and apply data-cleaning suggestions
- Ask Gemini for structured cleaning plans
- Generate analytics or transformation code plans with dry-run previews
- Generate chart plans and visualizations
- Record plans and executions in the backend audit trail

## Project Structure

```text
backend/
  app.py                 FastAPI application and API routes
  storage.py             In-memory data, plan, and audit storage
  services/
    ai.py                Gemini integration and plan validation
    cleaner.py           Data-cleaning operations
    executor.py          Restricted generated-code executor
frontend/
  app.py                 Streamlit user interface
tests/
  test_plan_validator.py Plan validation tests
requirements.txt         Python dependencies
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- A Google Gemini API key for AI-powered features

## Installation

From the project root, create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Configuration

Set the Gemini API key before starting the backend.

### Windows PowerShell

```powershell
$Env:GEMINI_API_KEY = "your-api-key"
```

Optional settings:

```powershell
$Env:GEMINI_MODEL = "models/gemini-2.5-flash"
$Env:GEMINI_TEMPERATURE = "0.1"
```

These environment variables apply only to the current terminal session. Set them again in any new terminal used to start the backend.

## Running the Application

The backend and frontend run as separate processes. Keep both terminals open.

### Terminal 1: FastAPI backend

```powershell
.venv\Scripts\activate
python -m uvicorn backend.app:app --reload
```

Backend URLs:

- API: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs

A `404 Not Found` response at `/` is expected because the application does not define a root route.

### Terminal 2: Streamlit frontend

```powershell
.venv\Scripts\activate
streamlit run frontend/app.py
```

Open the frontend at http://localhost:8501.

## Typical Workflow

1. Start both services.
2. Open the Streamlit frontend.
3. Upload a CSV file.
4. Generate a profile and review the dataset statistics.
5. Request cleaning suggestions or describe a task for Gemini.
6. Use dry-run previews before applying changes or executing generated transformations.
7. Generate charts from natural-language requests.

## Testing

Run the test suite from the project root:

```powershell
.venv\Scripts\activate
python -m pytest -q
```

## Troubleshooting

### `No module named ...`

Make sure the virtual environment is active and reinstall the dependencies:

```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Frontend cannot connect to the backend

Confirm that the FastAPI server is running at `http://127.0.0.1:8000` before using the Streamlit interface.

### Gemini features return an error

Confirm that `GEMINI_API_KEY` is set in the same terminal session used to start the backend.

### Port already in use

Start the backend or frontend on another port and update the frontend `BACKEND_URL` in `frontend/app.py` if the backend port changes.

## Notes

- Uploaded data and audit information are currently stored in memory and are lost when the backend restarts.
- Generated code should be reviewed with dry-run enabled before execution.
- The project currently uses the `google-generativeai` package for Gemini integration.
