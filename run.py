import subprocess
import sys
import time
import threading

def run_fastapi():
    """Run FastAPI server"""
    subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"])

def run_streamlit():
    """Run Streamlit app"""
    time.sleep(5)  # Wait for FastAPI to start
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])

if __name__ == "__main__":
    print("Starting Legal Document Search System...")
    print("FastAPI will be available at: http://localhost:8000")
    print("Streamlit will be available at: http://localhost:8501")
    
    # Start FastAPI in a separate thread
    api_thread = threading.Thread(target=run_fastapi)
    api_thread.daemon = True
    api_thread.start()
    
    # Start Streamlit in main thread
    run_streamlit()