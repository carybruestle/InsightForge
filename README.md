# InsightForge

Starter Python project for working with the sales dataset in `Data/sales_data.csv`.

## Project Structure

- `Data/` - input data files
- `main.py` - entry point script
- `dashboard.py` - Streamlit dashboard for interactive sales exploration
- `requirements.txt` - Python dependencies

## Quick Start

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Run the project:

   ```powershell
   python main.py
   ```

## Run The Dashboard

Start the interactive dashboard to explore sales by time period, region, product, and demographics:

```powershell
streamlit run dashboard.py
```

## Evaluate QA Quality And Efficiency

Run the offline QA evaluation runner with a gold dataset:

```powershell
python evals/run_qa_eval.py --dataset evals/datasets/qa_gold.jsonl
```

This generates:

- per-query JSONL results in `evals/reports/`
- a markdown summary report in `evals/reports/`

For setup and dataset schema details, see `evals/README.md`.
