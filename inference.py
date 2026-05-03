
import os
import io
import json
import joblib
import pandas as pd

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        return pd.read_csv(io.StringIO(request_body), header=None)
    elif request_content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    preds = model.predict(input_data)
    return preds

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction.tolist()), "application/json"
    return ",".join(map(str, prediction.tolist())), "text/csv"
