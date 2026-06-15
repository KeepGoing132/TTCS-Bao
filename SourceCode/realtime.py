import streamlit as st
import cv2
import av
import torch
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from PIL import Image

import config
from models import get_model
from dataset import get_raf_db_transforms

LABELS_CLEAN = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

def load_model():
    model = get_model(num_classes=config.NUM_CLASSES, pretrained=True)
    model.eval()
    return model


class EmotionProcessor(VideoProcessorBase):

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model().to(self.device)
        self.transform = get_raf_db_transforms(train=False)

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(gray, 1.1, 5, minSize=(50,50))

        for (x,y,w,h) in faces:

            face = img[y:y+h, x:x+w]

            try:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = Image.fromarray(face)

                tensor = self.transform(face).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self.model(tensor)
                    prob = torch.softmax(out, dim=1)
                    idx = torch.argmax(prob, dim=1).item()
                    conf = prob[0][idx].item()

                label = LABELS_CLEAN[idx]

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(img, f"{label} {conf*100:.1f}%",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

            except:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def run():

    st.subheader("📸 Realtime Emotion Detection")

    webrtc_streamer(
        key="emotion_cam",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )