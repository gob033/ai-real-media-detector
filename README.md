
# AI or Real Media Detector

A Streamlit web application for classifying photos and videos as AI-generated or real.

## Features

- Upload and analyze photos
- Upload and analyze videos
- CLIP ViT-B/32 image feature extraction
- Temporal pooling for video classification
- Display predicted class and model confidence
- Preview sampled video frames

## Models

- Photo: CLIP ViT-B/32 with a trained classifier
- Video: CLIP ViT-B/32 with temporal pooling

## Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Deployment

This project can be deployed using Streamlit Community Cloud connected to a GitHub repository.

## Disclaimer

This is a research prototype. Predictions are estimates and should not be treated as definitive proof of whether media is authentic.
