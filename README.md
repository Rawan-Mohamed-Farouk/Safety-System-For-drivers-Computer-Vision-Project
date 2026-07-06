# Safety-System-For-drivers-Computer-Vision-Project
# Safety-System-For-drivers-Computer-Vision-Project
A computer vision project focused on **driver safety monitoring**, using deep learning to detect unsafe driving behavior (such as drowsiness or distraction) in real time from camera input. This project represents **Phase 3** of the *"AI in Action: From Prediction to Safety"* series.
##Project Overview

This project aims to build a computer vision system that monitors a driver in real time and flags unsafe behavior — such as drowsiness, distraction, or improper attention to the road — to help reduce accident risk. It builds on earlier phases of the series (structured data prediction and image classification) by tackling a real-time, safety-critical vision task.

## Project Flow

| Step | Stage |
|------|-------|
| 1 | Data Collection / Upload |
| 2 | Data Preprocessing |
| 3 | Model Training (CNN / Computer Vision Model) |
| 4 | Evaluation |
| 5 | Deployment (Real-Time Detection App) |

---

## Dataset

- Input: images or video frames of drivers captured via camera
- Preprocessing steps typically include: face/eye detection, frame resizing, grayscale/normalization, and train-test splitting

## Model Pipeline

1. **Preprocessing** — face/eye region extraction, resizing, normalization
2. **Model Architecture** — Convolutional Neural Network (CNN) trained to classify driver state
3. **Training** — model fit on labeled driver-state image data
4. **Evaluation** — accuracy, precision/recall, and confusion matrix analysis on a held-out test set

## Deployment

The trained model can be deployed as a **real-time monitoring app** (e.g., via Streamlit, OpenCV webcam feed, or an edge device) that continuously analyzes driver footage and raises alerts when unsafe behavior is detected.

> *Add details about your deployment method (webcam demo, Streamlit app, alert mechanism such as sound/visual warning, etc.)*

---

## Tech Stack

- Python
- TensorFlow / Keras or PyTorch (CNN model)
- OpenCV (image/video processing, face & eye detection)
- Streamlit (optional, for web app deployment)

---

## Repository Structure

```
├── data/
│   ├── train/
│   └── test/
├── notebooks/
│   └── driver_safety_model.ipynb
├── app/
│   └── streamlit_app.py
├── models/
│   └── driver_safety_model.h5
├── requirements.txt
└── README.md
```
To try to test the app run a command like this in terminal
cli: python -m streamlit run (path_of_app.py)
