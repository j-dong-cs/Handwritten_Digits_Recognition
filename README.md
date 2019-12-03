# Handwritten_Digits_Recognition
A web service of recognizing digits on image.

- Build Image Preprocess Pipeline.
  - Convert to GrayScale
  
    ![org](/demo_images/original.png)
    ![gray](/demo_images/gray.png)
    
  - Apply Gaussian Blur
  
    ![org](/demo_images/original.png)
    ![blur](/demo_images/blur.png)
    
  - Apply threshold and invert colors
  
    ![org](/demo_images/original.png)
    ![thresh](/demo_images/thresh.png)
    
  - Use Canny Edge Detection to find edges
  
    ![org](/demo_images/original.png)
    ![edges](/demo_images/edges.png)
    
  - Dilate Edges found
  
    ![org](/demo_images/original.png)
    ![dilate](/demo_images/dilate.png)
    
  - Find Contours and bouding box for each digit
    ![boundingbox](/demo_images/boundingbox.png)
  - Extract each digit with boarder added to improve performance
  
    ![0](/demo_images/ROI_1.png)  ![1](/demo_images/ROI_1.png)  ![2](/demo_images/ROI_2.png)  ![3](/demo_images/ROI_3.png)  ![4](/demo_images/ROI_4.png)  ![5](/demo_images/ROI_5.png)

- Import MNIST digits data.

- Build and Train a Deep Neural Network Model to predict digits based on image pixels.
  - Convolutional Neural Network Pipeline:
    ![CNN pipeline](/demo_images/handwritten_digits_recognition_cnn.jpg)

  - Model performance
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

- Client convert image into bytes array and serialize it into JSON str.

- Client send a HTTP request with JSON str to digits recognition web service.

- The web service decode JSON str into bytes array which is sent to CNN model to predict digits on the image.

- The prediction of digits are returned to the client.
    ```python
    import json
    import base64

    image_path = os.path.join(os.getcwd(), 'test_images/IMG_7525.jpg')
    with open(image_path, 'rb') as file:
       img = file.read()
    image_64_encode = base64.encodebytes(img).decode('utf-8')
    bytes_to_json = json.dumps(image_64_encode)
       
    scoring_url = 'http://3ab34ad2-281d-4017-b47f-7a099895a46b.centralus.azurecontainer.io/score'
    headers = { 'Content-Type': 'application/json' }
    response = requests.post(scoring_url, input_data, headers=headers)
    print(json.loads(response.text))
    [0, 7, 1, 3, 8, 9]
    ```
