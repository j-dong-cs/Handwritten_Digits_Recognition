# Handwritten_Digits_Recognition
A web service of recognizing digits on image.

- Import MNIST digits data.

- Build and Train a Deep Neural Network Model to predict digits based on image pixels.
    ```python
    # calculate accuracy on the prediction
    acc = np.average(Y_pred == Y_true)
    print('Accuracy is', acc)
    Accuracy is 0.9912857142857143
    Attempted to log scalar metric accuracy:
    0.9912857142857143
    ```
- Store the DNN model into an ONNX:Open Neural Network Exchange instance.

- Load and Register the ONNX model into Azure.

- Create a Docker Image of the registered model, score.py script, and YAML envrionment dependencies file.

- Deploys the scoring image on Azure Containter Instance (ACI) as a web service.

- Client sends an HTTP POST request with image pixel data.

- The web service extracts pixel data from the JSON request.

- The pixle data is sent to the Keras Deep Learning pipeline model for scoring/prediction.

- The prediction of digits are returned to the client.
    ```python
    import json
    import requests
    with open('Nums.json') as json_file:
      json_data = json.load(json_file)
    input_data = json.dumps(json_data)
    scoring_url = "http://cddea800-aa7b-4ac7-aeae-ac3422fa8261.westus.azurecontainer.io/score"
    headers = { 'Content-Type': 'application/json' }
    response = requests.post(scoring_url, input_data, headers=headers)
    print(json.loads(response.text))
      [5,          1,         3,         3,         5,         5,         5,         7,         3,         2]
    ```
    ![alt text](nums.jpg)
