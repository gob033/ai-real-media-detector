

import io
import json
import os
import tempfile

import cv2
import numpy as np
import open_clip
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image


# ==========================================
# FAKTRION_033 BRANDING WITH CUSTOM LOGO
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(
    BASE_DIR, "assets", "faktrion_logo.png"
)

logo_col, title_col = st.columns(
    [1, 4],
    vertical_alignment="center"
)

with logo_col:
    if os.path.isfile(LOGO_PATH):
        st.image(LOGO_PATH, width=125)
    else:
        st.warning("Logo missing: assets/faktrion_logo.png")

with title_col:
    st.markdown(
        """
        <h1 style="
            font-size: 42px;
            font-weight: 800;
            margin: 0;
            padding: 0;
            color: #172033;
        ">FAKTRION_033</h1>
        """,
        unsafe_allow_html=True
    )

st.subheader("AI vs Real Photo & Video Detection")
st.caption("AI-Generated Media Detection System")

st.info(
    "Research prototype: predictions are estimates, "
    "not proof of authenticity."
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

PHOTO_HEAD_PATH = os.path.join(MODEL_DIR, "photo_clip_head.pth")
VIDEO_HEAD_PATH = os.path.join(MODEL_DIR, "video_clip_head.pth")
VIDEO_NORM_PATH = os.path.join(MODEL_DIR, "video_clip_norm.pt")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

DEVICE = torch.device("cpu")
torch.set_num_threads(2)

LABELS = ["AI-generated", "Real"]


# ==========================================
# 2. LOAD CLASSIFIER HEADS
# ==========================================
def load_head(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing model file: {path}"
        )

    bundle = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    # Keys exported by the Colab script are:
    # 0.weight, 0.bias, 3.weight, 3.bias
    head = nn.Sequential(
        nn.Linear(bundle["input_dim"], bundle["hidden_dim"]),
        nn.GELU(),
        nn.Dropout(0.25),
        nn.Linear(bundle["hidden_dim"], 2)
    )

    head.load_state_dict(bundle["state_dict"])
    head.eval()

    return head


# ==========================================
# 3. LOAD CLIP ONCE
# ==========================================
@st.cache_resource(show_spinner="Loading AI detection models...")
def load_models():
    required_files = [
        CONFIG_PATH,
        PHOTO_HEAD_PATH,
        VIDEO_HEAD_PATH,
        VIDEO_NORM_PATH
    ]

    for path in required_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    clip_model, _, preprocess = (
        open_clip.create_model_and_transforms(
            config["photo_encoder"],
            pretrained=config["photo_pretrained"],
            device="cpu"
        )
    )

    clip_model.eval()

    photo_head = load_head(PHOTO_HEAD_PATH)
    video_head = load_head(VIDEO_HEAD_PATH)

    norm = torch.load(
        VIDEO_NORM_PATH,
        map_location="cpu",
        weights_only=True
    )

    mean = norm["mean"].float().reshape(1, -1)
    std = norm["std"].float().reshape(1, -1).clamp_min(1e-5)

    return (
        clip_model,
        preprocess,
        photo_head,
        video_head,
        mean,
        std,
        config
    )


# ==========================================
# 4. PHOTO PREDICTION
# ==========================================
def predict_photo(image, clip_model, preprocess, head, config):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        features = clip_model.encode_image(tensor).float()

        # Must match the feature pipeline used in photo training.
        if config.get("photo_l2_normalize", False):
            features = torch.nn.functional.normalize(
                features, dim=1
            )

        logits = head(features)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_id = int(probabilities.argmax().item())

    return {
        "label": LABELS[predicted_id],
        "confidence": float(probabilities[predicted_id]),
        "probabilities": probabilities.tolist()
    }


# ==========================================
# 5. VIDEO FRAME EXTRACTION
# ==========================================
def extract_video_frames(video_bytes, extension, frame_count=8):
    temp_path = None
    cap = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp:
            temp.write(video_bytes)
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError(
                "Video open kora jayni. Onno MP4/MOV/AVI file try koro."
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            raise ValueError("Video-te readable frame paoa jayni.")

        indices = np.linspace(
            0,
            total_frames - 1,
            min(frame_count, total_frames),
            dtype=int
        )

        frames = []

        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = cap.read()

            if success and frame is not None:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
                frames.append(Image.fromarray(frame))

        if not frames:
            raise ValueError("Video theke frame extract kora jayni.")

        # Repeat frames for short videos.
        if len(frames) < frame_count:
            indices = np.linspace(
                0,
                len(frames) - 1,
                frame_count,
                dtype=int
            )
            frames = [frames[i] for i in indices]

        return frames[:frame_count], total_frames

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================
# 6. VIDEO PREDICTION
# ==========================================
def predict_video(
    frames,
    clip_model,
    preprocess,
    head,
    mean,
    std
):
    batch = torch.stack([
        preprocess(frame.convert("RGB"))
        for frame in frames
    ])

    with torch.inference_mode():
        frame_features = clip_model.encode_image(batch).float()

        # Match training: normalize each frame embedding first.
        frame_features = torch.nn.functional.normalize(
            frame_features,
            dim=1
        )

        mean_feature = frame_features.mean(dim=0)

        std_feature = frame_features.std(
            dim=0,
            unbiased=False
        )

        video_feature = torch.cat([
            mean_feature,
            std_feature
        ]).unsqueeze(0)

        # Apply saved training normalization.
        video_feature = (video_feature - mean) / std

        logits = head(video_feature)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_id = int(probabilities.argmax().item())

    return {
        "label": LABELS[predicted_id],
        "confidence": float(probabilities[predicted_id]),
        "probabilities": probabilities.tolist()
    }


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================
def show_result(result):
    if result["label"] == "AI-generated":
        st.warning("Prediction: AI-generated media")
    else:
        st.success("Prediction: Real media")

    st.metric(
        "Predicted class",
        result["label"]
    )

    st.metric(
        "Model confidence",
        f"{result['confidence'] * 100:.2f}%"
    )

    st.progress(result["confidence"])

    st.write("Class scores")
    st.write({
        LABELS[i]: f"{result['probabilities'][i] * 100:.2f}%"
        for i in range(2)
    })


# ==========================================
# 8. STREAMLIT USER INTERFACE
# ==========================================


st.write(
    "Upload a photo or video to classify it as AI-generated "
    "or real using trained CLIP-based classifiers."
)

st.info(
    "Research prototype: predictions are estimates, not proof "
    "of authenticity. Confidence is not a guarantee of correctness."
)

try:
    (
        clip_model,
        preprocess,
        photo_head,
        video_head,
        video_mean,
        video_std,
        config
    ) = load_models()

    st.caption("Models loaded successfully.")

except Exception as error:
    st.error("Model loading failed.")
    st.code(str(error))
    st.stop()


photo_tab, video_tab, about_tab = st.tabs([
    "🖼️ Photo Detection",
    "🎬 Video Detection",
    "ℹ️ About"
])


# ==========================================
# PHOTO TAB
# ==========================================
with photo_tab:
    st.subheader("AI Photo vs Real Photo")

    photo_file = st.file_uploader(
        "Choose a photo",
        type=["jpg", "jpeg", "png", "webp", "avif"],
        key="photo_upload"
    )

    if photo_file is not None:
        try:
            image = Image.open(photo_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded photo",
                use_container_width=True
            )

            if st.button(
                "Analyze Photo",
                type="primary",
                key="analyze_photo"
            ):
                with st.spinner("Analyzing photo..."):
                    result = predict_photo(
                        image,
                        clip_model,
                        preprocess,
                        photo_head,
                        config
                    )

                show_result(result)

        except Exception as error:
            st.error(f"Photo analysis failed: {error}")


# ==========================================
# VIDEO TAB
# ==========================================
with video_tab:
    st.subheader("AI Video vs Real Video")

    video_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi"],
        key="video_upload"
    )

    if video_file is not None:
        video_bytes = video_file.getvalue()

        st.write("File:", video_file.name)
        st.write(
            f"Size: {len(video_bytes) / (1024 * 1024):.2f} MB"
        )

        if st.button(
            "Analyze Video",
            type="primary",
            key="analyze_video"
        ):
            if len(video_bytes) > 150 * 1024 * 1024:
                st.error("Please upload a video smaller than 150 MB.")

            else:
                try:
                    with st.spinner(
                        "Extracting frames and analyzing video..."
                    ):
                        frames, total_frames = extract_video_frames(
                            video_bytes,
                            os.path.splitext(video_file.name)[1].lower(),
                            frame_count=int(config["video_frames"])
                        )

                        result = predict_video(
                            frames,
                            clip_model,
                            preprocess,
                            video_head,
                            video_mean,
                            video_std
                        )

                    st.write(
                        f"Frames in original video: {total_frames}"
                    )
                    st.write(
                        f"Frames analyzed: {len(frames)}"
                    )

                    st.image(
                        frames,
                        caption=[
                            f"Sample frame {i + 1}"
                            for i in range(len(frames))
                        ],
                        width=140
                    )

                    show_result(result)

                except Exception as error:
                    st.error(f"Video analysis failed: {error}")


# ==========================================
# ABOUT TAB
# ==========================================
with about_tab:
    st.subheader("About this project")

    st.write(
        "This research application uses CLIP ViT-B/32 embeddings "
        "and trained classifiers for AI-generated versus real media."
    )

    st.write("Photo model: CLIP ViT-B/32 + classifier")
    st.write("Video model: CLIP ViT-B/32 + temporal pooling")
    st.write(
        f"Video frames sampled per prediction: "
        f"{config['video_frames']}"
    )

    st.warning(
        "Performance may differ on unseen generators, edited media, "
        "compression artifacts, or media outside the training dataset."
    )

st.divider()
st.caption("AI vs Real Media Detection • Research prototype")
