from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load the object detection model and labels
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' in request.files:
        # Handle image upload
        image = request.files['image']

        # Read and decode the image
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Get the height and width of the input image
        height, width = img.shape[:2]

        # Determine the aspect ratio of the image
        aspect_ratio = width / height

        # Determine the new dimensions while maintaining the aspect ratio
        new_width = 320  # Or your desired width
        new_height = int(new_width / aspect_ratio)

        # Resize the image to the new dimensions
        img = cv2.resize(img, (new_width, new_height))

        # Set the adjusted input size
        model.setInputSize(new_width, new_height)
        model.setInputScale(1.0 / 127.5)
        model.setInputMean((127.5, 127.5, 127.5))
        model.setInputSwapRB(True)

        # Perform object detection
        ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

        # Process the object detection results
        results = []
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            label = classLabels[ClassInd - 1] if ClassInd <= len(classLabels) else 'Unknown'
            results.append({'label': label, 'confidence': float(conf), 'box': boxes.tolist()})

        # Draw bounding boxes on the image
        for box in results:
            label = box['label']
            (startX, startY, endX, endY) = box['box']
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert the image to JPEG format for display on the web
        _, image_data = cv2.imencode(".jpg", img)

        # Send the image as a response to be displayed on the web
        return Response(image_data.tobytes(), content_type="image/jpeg")

if __name__ == '__main__':
    app.run(debug=True)
