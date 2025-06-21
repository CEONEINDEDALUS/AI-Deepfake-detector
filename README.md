# AI Deepfake Detector


https://github.com/user-attachments/assets/f30e38ae-4046-4b43-847a-b11620a73953


A desktop application for detecting deepfake faces in videos using deep learning and computer vision. The app features a user-friendly GUI built with PyQt5 and leverages OpenCV, MediaPipe, and a CNN-based model for real-time detection. Users can label detected faces as real or fake for feedback, and fine-tune the detection model with their feedback data.

## Features

- **Video Analysis**: Load and play video files, and analyze frames for deepfake faces.
- **Deepfake Detection**: Uses a convolutional neural network (CNN) to detect deepfakes.
- **Facial Landmarking**: Utilizes MediaPipe FaceMesh for accurate face localization.
- **Artifact Detection**: Employs blob detection to highlight potential artificial artifacts.
- **Blink Detection**: Auxiliary blink detection for additional signals.
- **Interactive GUI**: Intuitive controls for loading, playing, pausing, and stopping videos.
- **Feedback Loop**: Mark detected faces as real or fake to build a feedback dataset.
- **Model Fine-Tuning**: Fine-tune the detection model using labeled feedback data.
- **Adjustable Threshold**: Set the detection sensitivity threshold.

### Prerequisites

- Python 3.7+
- [PyQt5](https://pypi.org/project/PyQt5/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [mediapipe](https://pypi.org/project/mediapipe/)
- [torch](https://pypi.org/project/torch/)
- [numpy](https://pypi.org/project/numpy/)
- [requests](https://pypi.org/project/requests/)


### Download/Prepare Model

The application attempts to download a pretrained model (`deepfake_model.pth`) on first run.  
**Note:** Update the model download URL in the code (`VideoProcessor.download_model`) with an actual model link or place your model file in the app directory.

## Usage

1. **Run the App**

    ```bash
    python deepfakedetector.py
    ```

2. **Load a Video**

    - Click "Load Video" and select a video file (`.mp4`, `.avi`, `.mov`).

3. **Choose Mode**

    - `Simple Player`: Just plays the video.
    - `Deepfake Finder`: Analyzes for deepfake faces.

4. **Play/Pause/Stop**

    - Use Play, Pause, and Stop buttons to control playback.

5. **Set Threshold**

    - Adjust the detection threshold slider to set sensitivity.

6. **Label Faces**

    - When faces are detected, label them as "Real" or "Fake" for feedback.

7. **Fine-Tune Model**

    - Click "Fine-Tune Model" to retrain the detector using your labeled data.

## Feedback Data

Labeled faces are stored in a local SQLite database (`feedback.db`).  
Fine-tuning uses this dataset to improve detection.

## Notes

- The demo model is a placeholder; for best results, use a real pretrained deepfake detection model.
- For performance, a CUDA-enabled GPU is recommended but not required.

## File Structure

- `deepfakedetector.py` — Main application file containing all logic, model, and UI.

## License

MIT License

## Acknowledgments

- [PyQt5](https://riverbankcomputing.com/software/pyqt/intro)
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [Torch](https://pytorch.org/)
