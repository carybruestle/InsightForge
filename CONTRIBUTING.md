# Contributing to InsightForge

Thanks for helping improve InsightForge.

## Development setup

1. Clone the repository.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Copy the sample environment file and add your local configuration:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Update the values in `.env` for your local environment.

## Running the project

```powershell
python main.py
streamlit run dashboard.py
```

## Quality expectations

- Keep changes focused and scoped.
- Prefer readable, maintainable code over clever shortcuts.
- Document any new environment variables or setup steps.
- Validate the relevant workflow before submitting a change.

## Pull requests

- Create a branch for your work.
- Keep commit messages clear and specific.
- Include context in the PR description about what changed and why.

## Troubleshooting

If the dashboard or evaluator does not run as expected, confirm:

- the Python environment is active,
- dependencies are installed,
- all required environment variables are configured in `.env`.
