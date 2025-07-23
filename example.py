from flask import Flask, request, render_template_string, jsonify
from PIL import Image, ImageDraw, ImageOps
import io
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('path/to/your_model.h5')  # Update with your model path
print("Model loaded successfully")

# Image preprocessing
def preprocess_image(img):
    img = img.resize((28, 28)).convert('L')  # Resize and convert to grayscale
    img = ImageOps.invert(img)  # Invert colors (MNIST style)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

class PhraseRecognizer:
    def __init__(self):
        self.reset_canvas()
        self.current_phrase = []
        
    def reset_canvas(self):
        self.canvas_img = Image.new("RGB", (800, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.canvas_img)
        
    def add_stroke(self, points):
        self.draw.line(points, fill="black", width=8)
        
    def recognize_phrase(self):
        # Word segmentation using vertical projection
        img_gray = np.array(self.canvas_img.convert('L'))
        vertical_projection = np.sum(img_gray < 200, axis=0)  # Threshold for "ink"
        
        # Find word boundaries
        in_word = False
        word_boundaries = []
        for i, val in enumerate(vertical_projection):
            if val > 5 and not in_word:  # Word starts
                in_word = True
                word_boundaries.append(i)
            elif val <= 5 and in_word:  # Word ends
                in_word = False
                word_boundaries.append(i)
        
        # Handle case where drawing ends mid-word
        if in_word:
            word_boundaries.append(len(vertical_projection)-1)
            
        #Process each word
        recognized_phrase = []
        for i in range(0, len(word_boundaries), 2):
            if i+1 >= len(word_boundaries):
                break
                
            left = word_boundaries[i]
            right = word_boundaries[i+1]
            word_img = self.canvas_img.crop((left, 0, right, 200))
            
            # Step 3: Letter segmentation within each word
            word_gray = np.array(word_img.convert('L'))
            horizontal_projection = np.sum(word_gray < 200, axis=1)
            
            # Find letter boundaries (simplified)
            letters = []
            letter_start = 0
            for j in range(1, len(horizontal_projection)):
                if horizontal_projection[j] > 5 and horizontal_projection[j-1] <= 5:
                    letter_start = j
                elif horizontal_projection[j] <= 5 and horizontal_projection[j-1] > 5:
                    letter_img = word_img.crop((0, letter_start, word_img.width, j))
                    letters.append(letter_img)
            
            # Recognize each letter
            word_text = []
            for letter_img in letters:
                # Preprocess and predict
                processed = preprocess_image(letter_img)
                predictions = model.predict(processed)
                predicted_class = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions)
                
                # Convert to character (A=0, B=1, ... Z=25)
                predicted_char = chr(65 + predicted_class)
                word_text.append(predicted_char)
            
            recognized_phrase.append(''.join(word_text))
        
        return ' '.join(recognized_phrase)

recognizer = PhraseRecognizer()

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Handwriting Recognition</title>
            <style>
                #canvas {
                    border: 2px solid #333;
                    background: white;
                    cursor: crosshair;
                }
                #prediction {
                    font-size: 2em;
                    margin: 20px;
                    padding: 10px;
                    border: 1px dashed #aaa;
                    min-height: 1.2em;
                }
                button {
                    padding: 10px 20px;
                    margin: 5px;
                    font-size: 1em;
                }
            </style>
        </head>
        <body>
            <h1>Draw a Phrase</h1>
            <canvas id="canvas" width="800" height="200"></canvas>
            <div>
                <button onclick="clearCanvas()">Clear</button>
                <button onclick="recognize()">Recognize</button>
            </div>
            <div id="prediction"></div>

            <script>
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');
                const predictionDiv = document.getElementById('prediction');
                let isDrawing = false;
                let lastX = 0;
                let lastY = 0;
                
                // Initialize white background
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.strokeStyle = 'black';
                ctx.lineWidth = 8;
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                
                canvas.addEventListener('mousedown', startDrawing);
                canvas.addEventListener('mousemove', draw);
                canvas.addEventListener('mouseup', stopDrawing);
                canvas.addEventListener('mouseout', stopDrawing);
                
                function startDrawing(e) {
                    isDrawing = true;
                    [lastX, lastY] = [e.offsetX, e.offsetY];
                }
                
                function draw(e) {
                    if (!isDrawing) return;
                    
                    ctx.beginPath();
                    ctx.moveTo(lastX, lastY);
                    ctx.lineTo(e.offsetX, e.offsetY);
                    ctx.stroke();
                    
                    // Send drawing data to server
                    fetch('/draw', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            points: [[lastX, lastY], [e.offsetX, e.offsetY]]
                        })
                    });
                    
                    [lastX, lastY] = [e.offsetX, e.offsetY];
                }
                
                function stopDrawing() {
                    isDrawing = false;
                }
                
                function clearCanvas() {
                    ctx.fillStyle = 'white';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    fetch('/clear', {method: 'POST'});
                    predictionDiv.textContent = '';
                }
                
                function recognize() {
                    fetch('/recognize', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => {
                            predictionDiv.textContent = data.phrase || "No text recognized";
                        });
                }
            </script>
        </body>
        </html>
    ''')

@app.route('/draw', methods=['POST'])
def handle_draw():
    data = request.json
    recognizer.add_stroke(data['points'])
    return jsonify({'status': 'ok'})

@app.route('/clear', methods=['POST'])
def handle_clear():
    recognizer.reset_canvas()
    return jsonify({'status': 'cleared'})

@app.route('/recognize', methods=['POST'])
def handle_recognize():
    try:
        phrase = recognizer.recognize_phrase()
        return jsonify({'phrase': phrase, 'status': 'success'})
    except Exception as e:
        return jsonify({'phrase': f"Error: {str(e)}", 'status': 'error'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)