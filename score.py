from azureml.core.model import Model
import json
import io
import numpy as np
import pandas as pd
import keras
import onnxruntime

def init():
    global model_path, session, input_name, output_name
    model_path = Model.get_model_path(model_name="onnxmodelimage")
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
# note you can pass in multiple rows for scoring
def run(raw_data):
    img_cols = 28
    img_rows = 28
    try:
        #with open(raw_data) as json_file:
        #    data = json.load(json_file)
        data = np.array(json.loads(raw_data)).astype('float32')
        data = data.reshape(data.shape[0], img_rows, img_cols, 1)
        r = session.run([output_name], {input_name: data})
        for row in r: # select the indix with the maximum probability
            result = pd.Series(np.array(row).argmax(axis=1), name="Label")
        output = io.StringIO()
        json.dump(result.tolist(), output)
        return output.getvalue()
    except Exception as e:
        error = str(e)
        return error
