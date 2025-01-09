import pandas as pd
import numpy as np
import time
import sys
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
from src.utils.utils import fetch_path_from_config
from src.constants.constants import BASE_DIR,CONFIG_PATH

data_path = fetch_path_from_config("Paths", "data_path", CONFIG_PATH)

raw_df = pd.read_csv(data_path, index_col=0)
raw_df.head()

final_df = raw_df

latest_record = None

def generate_data(iter_index):
    global latest_record
    while True:
        result_data =  final_df.loc[next(iter_index)].to_json()
        latest_record = result_data
        yield f"{result_data}\n"
        time.sleep(60)

iter_index = iter(final_df.index.to_list())
        

app = FastAPI()

data_stream = None

@app.get("/generate")
def start_api():

    global data_stream

    if data_stream is None:
        data_stream = generate_data(iter_index)
    return StreamingResponse(data_stream, media_type="application/json")

@app.get("/stop")
def stop_data_generation():
    """Stop real-time data generation."""
    global data_stream
    data_stream = None
    return {"message": "Data generation stopped"}
    
@app.get("/")
def read_root():
    """API Root Endpoint."""
    return {"message": "Welcome to the Real-Time Data API"}
        
@app.get("/latest")
def get_latest_record():
    global latest_record
    if latest_record:
        return json.loads(latest_record)  # Return the latest record as a JSON response
    else:
        return {"message": "No data has been generated yet."}

# uvicorn generate_data:app --reload --host 0.0.0.0 --port 8000