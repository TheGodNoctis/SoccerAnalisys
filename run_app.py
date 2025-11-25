import webview
import subprocess
import threading
import os

def start_streamlit():
    subprocess.Popen(["streamlit", "run", "app.py", "--server.headless=true"])

threading.Thread(target=start_streamlit, daemon=True).start()

# Abre a janela com a interface
webview.create_window(
    "⚽ Análise Tática de Futebol",
    "http://localhost:8501",
    width=1280,
    height=800,
    resizable=True
)
webview.start()
