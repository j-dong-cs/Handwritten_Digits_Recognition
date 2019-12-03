from azureml.core.model import Model
import json
import io
import numpy as np
import pandas as pd
import cv2 as cv
import keras
import onnxruntime
import base64

def init():
    global model_path, session, input_name, output_name
    model_path = Model.get_model_path(model_name="onnxmodelimage")
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

# This method preprocess input image: image -> grayscale -> blur -> threshold -> edges -> dilate 
# in order to make it ready to be passed to model for prediction
def preprocess(image):
    global thresh, contours
    # resize original image to be fixed size 640 x 480
    image = cv.resize(image, (640, 480))
    # convert image to gray scale of pixel value from 0 to 255
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # apply gaussian blur to filter image
    blur = cv.GaussianBlur(gray,(5,5),0)
    # apply threshold on blurred image to amplify digits
    ret,thresh = cv.threshold(blur, 120, 200, cv.THRESH_BINARY_INV)    
    # find digits edges using Canny Edge Detection
    edges = cv.Canny(thresh, 120, 200)
    # apply dilation on detected edges
    kernel = np.ones((4,4),np.uint8)
    dilate = cv.dilate(edges, kernel)
    
    # find contours and get the external one
    contours, hier = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(type(contours))

# This method find the bounding box for each digit in the image based on contours
def findBoundingBoxes():
    rect = []
    # with each contour, draw boundingRect in blue
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv.boundingRect(c)
        rect.append([x, y, w, h])
    print(type(rect))
    return rect

# This method merge bounding boxes for same digit
# and sort each box by x-axis value
def mergeBoundingBoxes(rect):
    rect = np.array(rect)
    for i in range(len(rect)):
        j = i + 1
        while j < len(rect):
            # check if rect[j] is within rect[i]
            xBound = rect[j][0] > rect[i][0] and rect[j][0]+rect[j][2] < rect[i][0]+rect[i][2]
            yBound = rect[j][1] > rect[i][1] and rect[j][1]+rect[j][3] < rect[i][1]+rect[i][3]
            if (xBound and yBound) == True:
                rect = np.delete(rect, j, 0)
                j = i + 1
            else:
                j = j + 1
    # sort bounding boxes on x-axis value
    groupedRect = rect[rect[:,0].argsort()].tolist()
    print(type(groupedRect))
    return groupedRect

# This method iterate thorugh bounding boxes and extract for ROI
def extractROI(rect):
    digits = []
    original = thresh.copy()
    image_number = 0
    for pts in rect:
        # add border to each digit
        ROI = original[pts[1]-20:pts[1]+pts[3]+20, pts[0]-20:pts[0]+pts[2]+20]
        digits.append(ROI)
        # cv.imwrite("ROI_{}.png".format(image_number), ROI)
        image_number += 1
    print(type(digits))
    return digits

# This method resize each digit image to be 28 x 28 and normalize its values to be between 0 to 1
def resizeAndNormalize(digits):
    input_data = []
    for digit in digits:
        digit = cv.resize(digit, (28,28))
        digit = np.divide(digit, 255)
        input_data.append(digit)
    print(type(input_data))
    return input_data
        
# note you can pass in multiple rows for scoring
def run(raw_data):
    img_cols = 28
    img_rows = 28
    try:
        #with open(raw_data) as json_file:
        #    data = json.load(json_file)
        
        # convert JSON format img str to bytes and decode back to img file
        json_to_bytes = json.loads(raw_data).encode('utf-8')
        decoded_img = base64.decodebytes(json_to_bytes)
        image = cv.imdecode(np.asarray(bytearray(decoded_img), dtype="uint8"), cv.IMREAD_COLOR)  
        
        #image_name = 'imgToPred.jpg'
        #with open(image_name, 'wb') as image_result:
        #    image_result.write(decoded_img)        
        #image_path = os.path.join(os.getcwd(), image_name)
        
        preprocess(image)
        rect = findBoundingBoxes()
        groupedRect = mergeBoundingBoxes(rect)
        digits = extractROI(groupedRect)
        input_data = resizeAndNormalize(digits)
        
        input_data = np.array(input_data).astype('float32')
        input_data = input_data.reshape(input_data.shape[0], img_rows, img_cols, 1)
        r = session.run([output_name], {input_name: input_data})
        for row in r: # select the indix with the maximum probability
            result = pd.Series(np.array(row).argmax(axis=1), name="Label")
        output = io.StringIO()
        json.dump(result.tolist(), output)
        return output.getvalue()
    except Exception as e:
        error = str(e)
        return error
