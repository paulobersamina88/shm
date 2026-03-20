# RS4D Near-Real-Time SHM Dashboard

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Summary
This Streamlit app reads ENE, ENN, and EHZ CSV files, computes dominant frequencies, and displays:
- 🟢 Building Status: NORMAL
- 🟡 WARNING: Frequency Shift Detected
- 🔴 ALERT: Possible Structural Change
