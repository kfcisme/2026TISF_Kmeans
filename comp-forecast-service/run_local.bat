@echo off
set PORT=8900
set USE_ONNX=false
set USE_TORCH=false
uvicorn app:app --host 0.0.0.0 --port %PORT%
