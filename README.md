# License Plate Recognition System

This project is a license plate recognition system that can detect and recognize license plates from video files and images.

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command-Line Interface

To run the license plate recognition system on a video file from the command line, use the following command:

```bash
python src/main.py --video <path_to_video> --output <path_to_output_directory>
```

#### Arguments

*   `--video`: Path to the input video file. (Default: `data/videos/plate_test.mp4`)
*   `--output`: Path to the directory to save the output CSV file. (Default: `results/`)

#### Example

```bash
python src/main.py --video data/videos/nepali.mp4 --output results/
```

### Frontend Application

The frontend application allows you to recognize license plates from both images and videos through a web interface.

To run the frontend application, navigate to the `frontend` directory and run the `app.py` script:

```bash
cd frontend
python app.py
```

Then, open your web browser and go to `http://127.0.0.1:5000` to use the application. You can upload an image or a video file to recognize the license plates.
