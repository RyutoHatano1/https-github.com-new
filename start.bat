@echo off
start powershell -NoExit -Command "cd backend; uvicorn main:app --reload"
timeout /t 5
start powershell -NoExit -Command "cd frontend; python -m http.server 3000"
start http://localhost:3000
