from flask import Flask, render_template, request, jsonify
import os
import easyocr

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

reader = easyocr.Reader(['en'], gpu=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result = reader.readtext(filepath, detail=0)
    return jsonify({'plate': ' | '.join(result)})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
