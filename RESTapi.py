import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('FRSv_croppedv1.1.keras')

model.summary()

# Load the class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

def preprocess_image(image, target_size=(500, 250)):
    """
    Resize the image to the target size and normalize it.
    """
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize the image to the range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the postman request
        image_file = request.files['image']
        image = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Preprocess the image same like what we used during model
        preprocessed_image = preprocess_image(image, target_size=(500, 250))
        
        # Get predictions from the model
        predictions = model.predict(preprocessed_image)
        
        # Get the index of the highest prediction, top first element
        predicted_index = np.argmax(predictions, axis=1)[0]
        
        # Get the predicted class name based on index.
        predicted_class = class_names[predicted_index]
        
        # Get the matching score (probability)
        matching_score = predictions[0][predicted_index]
        
        # Return the predicted class name and matching score as a JSON response
        return jsonify({
            'predicted_class': predicted_class,
            'matching_score': float(matching_score)  # Convert to float for JSON serialization
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
