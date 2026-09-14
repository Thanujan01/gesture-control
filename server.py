"""
Gesture detection server for AMB82-Mini — FINAL.

Receives a JPEG via HTTP POST from the board, runs the custom-trained
MediaPipe gesture recognizer, returns the detected gesture as JSON.

ENDPOINTS
---------
  POST /detect   board posts a JPEG, gets {"gesture": ..., "score": ...}
  GET  /health   {"status": "ok"}

MODEL
-----
  Two-stage: MediaPipe's hand detector finds 21 landmarks, features.py turns
  them into 118 orientation-independent features, then your trained
  scikit-learn classifier (gesture_model.pkl) labels the hand shape.

  features.py is shared with extract_landmarks.py so training and inference
  can never drift apart — that mismatch silently produces garbage predictions.

  Needs BOTH files in this folder:
    gesture_recognizer.task  — stock MediaPipe download, hand detection only
    gesture_model.pkl        — from train_classifier.py, your 11 classes

  MediaPipe Model Maker was the original plan but no longer installs on
  current Colab (tensorflow-addons has no Python 3.12 wheels). The landmark
  approach trains locally in under a minute and is just as accurate.

  Labels: five, closed_fist, one, two, thumbs_up, thumb_down,
          best, call, super, three, none

  Note these are lowercase, unlike the pre-trained model's 'Open_Palm' style.
  GESTURE_NAMES[] in the sketch must match exactly, case included.

  The model's own 'none' class is mapped to "None" in the response, so the
  board treats it the same as no hand found.

DETECTION
---------
  Scans up to TWO hands and keeps whichever gesture scores highest. Single-hand
  detection missed the gesturing hand when the other was partly in frame,
  because the palm detector locked onto the wrong one.

  Every frame is logged with what was seen, including scores below
  MIN_CONFIDENCE, so a rejected gesture can be told apart from a hand that was
  never found:

      1 hand(s)  Right:two=0.94        -> SENT two
      1 hand(s)  Left:three=0.41       -> sent None
      no hand found                    -> sent None

TUNING
------
  MIN_CONFIDENCE is the main dial. A custom-trained model scores much higher
  on its own classes than the generic one did, so 0.70 is a reasonable start.
  Raise it if you get flicker between similar gestures; lower it if detection
  is intermittent. The logs show the real numbers to decide with.

RUN
---
  uvicorn server:app --host 0.0.0.0 --port 8000

TEST (PowerShell — use curl.exe, not curl)
------------------------------------------
  curl.exe -X POST --data-binary "@test.jpg" -H "Content-Type: image/jpeg" http://localhost:8000/detect
"""

import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from features import extract
from fastapi import FastAPI, Request

MODEL_PATH = "gesture_recognizer.task"   # used only for hand detection
CLASSIFIER_PATH = "gesture_model.pkl"    # your trained classifier
MIN_CONFIDENCE = 0.70

app = FastAPI()

# MediaPipe finds the hand and its 21 landmarks
recognizer = vision.GestureRecognizer.create_from_options(
    vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
)

# Your classifier turns those landmarks into one of the 11 labels
_bundle = joblib.load(CLASSIFIER_PATH)
clf, LABELS = _bundle["model"], _bundle["labels"]
print(f"Loaded classifier with {len(LABELS)} classes: {', '.join(LABELS)}")



@app.post("/detect")
async def detect(request: Request):
    jpeg_bytes = await request.body()

    # Decode JPEG -> BGR array -> RGB (MediaPipe expects RGB)
    bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"gesture": "None", "score": 0.0, "error": "bad_jpeg"}

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    # Classify EVERY detected hand, keep whichever scores highest
    label, score = "None", 0.0
    parts = []
    for i, landmarks in enumerate(result.hand_landmarks):
        # Handedness drives the left/right collapse in features.extract, so it
        # must be read BEFORE extracting rather than only used for logging.
        side = "Right"
        if i < len(result.handedness):
            side = result.handedness[i][0].category_name

        feats = extract(landmarks, side).reshape(1, -1)
        probs = clf.predict_proba(feats)[0]
        best = int(np.argmax(probs))
        name, conf = LABELS[best], float(probs[best])

        parts.append(f"{side}:{name}={conf:.2f}")

        if conf > score:
            score = conf
            # 'none' means "a hand, but not one of the 10" — same outcome
            # for the board as nothing found at all.
            if conf >= MIN_CONFIDENCE and name != "none":
                label = name
            else:
                label = "None"

    if not result.hand_landmarks:
        detail = "no hand found"
    else:
        detail = f"{len(result.hand_landmarks)} hand(s)  " + "  ".join(parts)

    print(detail + (f"   -> SENT {label}" if label != "None" else "   -> sent None"))
    return {"gesture": label, "score": round(score, 3)}


@app.get("/health")
def health():
    return {"status": "ok"}
