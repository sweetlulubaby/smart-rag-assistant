$env:LANGCHAIN_TRACING_V2 = "false"
$env:LANGSMITH_TRACING = "false"

Write-Host "Starting demo on http://localhost:8501"
Write-Host "If first retrieval is slow, it is likely loading embeddings/reranker into memory."

& ".\.venv\Scripts\streamlit.exe" run "app.py" --server.headless true --server.port 8501
