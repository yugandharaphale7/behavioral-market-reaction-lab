py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m compileall src run_pipeline.py
Write-Host 'Setup complete. Put analyst_ratings_processed.csv in data\raw, then run: python run_pipeline.py'
