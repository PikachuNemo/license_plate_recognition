import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
from run import main as run_main

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process the video using the main function from run.py
    output_dir = 'results/'
    output_video_path = os.path.join(output_dir, 'output_video.mp4')
    interpolated_csv_path = run_main(filepath, output_dir, output_video_path)

    if interpolated_csv_path:
        return jsonify({'plate': f'Processing complete. Results saved to {interpolated_csv_path}'})
    else:
        return jsonify({'error': 'Failed to process video'}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

