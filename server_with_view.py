"""
Gesture detection server WITH live view — matches server.py's detection.

Identical detection behaviour to server.py (MediaPipe landmarks + your trained
classifier), plus a browser view so you can see what the camera sees.

ENDPOINTS
---------
  POST /detect   board posts a JPEG, gets {"gesture": ..., "score": ...}
  GET  /view     live annotated feed, self-refreshing
  GET  /latest   most recent annotated frame (JPEG)
  GET  /stream   MJPEG stream (fallback; /view no longer uses it)
  GET  /health   {"status": "ok"}

WHEN TO USE THIS INSTEAD OF server.py
-------------------------------------
  When detection misbehaves. The logs tell you the scores; the view tells you
  WHY — hand out of frame, too dark, too far, or tracking fine but classifying
  wrong all look identical in a log and obvious on screen.

  Only one server can hold port 8000. Run this OR server.py, never both:
    Get-NetTCPConnection -LocalPort 8000 -State Listen |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

NEEDS
-----
  gesture_recognizer.task  — stock MediaPipe download, hand detection only
  gesture_model.pkl        — from train_classifier.py, your 11 classes

RUN
---
  uvicorn server_with_view:app --host 0.0.0.0 --port 8000
  then open  http://192.168.1.129:8000/view

  The feed only updates while the board is POSTing, so start the sketch too.
  ~2 fps is expected — that's the board's send rate, not a viewer limit.
"""

import threading
import time

import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from features import extract
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, Response

MODEL_PATH = "gesture_recognizer.task"   # used only for hand detection
CLASSIFIER_PATH = "gesture_model.pkl"    # your trained classifier
MIN_CONFIDENCE = 0.70

# Labels meaning "a hand, but not one of the real gestures". Mapped to "None"
# so the board treats them like no hand found. Must match whatever the
# negative class is actually called, or the board's re-arm logic stalls.
NO_GESTURE_LABELS = {"none", "no_gesture", "nogesture", "negative", "other"}

# MediaPipe's 21-point hand skeleton, as index pairs
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (9, 10), (10, 11), (11, 12),              # middle
    (13, 14), (14, 15), (15, 16),             # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm
]

app = FastAPI()

recognizer = vision.GestureRecognizer.create_from_options(
    vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
)

_bundle = joblib.load(CLASSIFIER_PATH)
clf, LABELS = _bundle["model"], _bundle["labels"]
print(f"Loaded classifier with {len(LABELS)} classes: {', '.join(LABELS)}")

# Most recent annotated frame, shared between the POST handler and the viewer
_latest_jpeg = None
_lock = threading.Lock()



def annotate(bgr, result, label, score, detail):
    """Draw every detected hand skeleton plus diagnostic text."""
    out = bgr.copy()
    h, w = out.shape[:2]

    for hand_idx, landmarks in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(out, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(out, p, 4, (0, 0, 255), -1)

        side = "?"
        if hand_idx < len(result.handedness):
            side = result.handedness[hand_idx][0].category_name
        if pts:
            cv2.putText(out, side, (pts[0][0] - 20, pts[0][1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    colour = (0, 200, 0) if label != "None" else (0, 0, 220)
    text = f"{label}  {score:.2f}" if label != "None" else "No gesture"

    cv2.rectangle(out, (0, 0), (w, 62), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(out, detail, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)
    return out


@app.post("/detect")
async def detect(request: Request):
    global _latest_jpeg

    jpeg_bytes = await request.body()
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
            # A negative-class hit means "a hand, but not a gesture" — same
            # outcome for the board as nothing found at all.
            if conf >= MIN_CONFIDENCE and name.lower() not in NO_GESTURE_LABELS:
                label = name
            else:
                label = "None"

    h, w = bgr.shape[:2]
    size = f"{w}x{h} {len(jpeg_bytes)//1024}kB"
    if not result.hand_landmarks:
        detail = f"no hand found   [{size}]"
    else:
        detail = (f"{len(result.hand_landmarks)} hand(s)  "
                  + "  ".join(parts) + f"   [{size}]")

    annotated = annotate(bgr, result, label, score, detail)
    ok, encoded = cv2.imencode(".jpg", annotated)
    if ok:
        with _lock:
            _latest_jpeg = encoded.tobytes()

    print(detail + (f"   -> SENT {label}" if label != "None" else "   -> sent None"))
    return {"gesture": label, "score": round(score, 3)}


@app.get("/latest")
def latest():
    with _lock:
        frame = _latest_jpeg
    if frame is None:
        return Response(content=b"", media_type="image/jpeg", status_code=404)
    return Response(content=frame, media_type="image/jpeg")


@app.get("/view", response_class=HTMLResponse)
def view():
    """Live view via JS polling of /latest — more reliable than MJPEG."""
    return """
    <html>
      <head>
        <title>AMB82-Mini Gesture View</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
      </head>
      <body style="background:#111;color:#eee;font-family:sans-serif;
                   text-align:center;margin:0;padding:16px">
        <h2 style="margin:8px 0">AMB82-Mini Live View</h2>
        <div style="margin:8px 0;font-size:14px">
          <span id="status" style="color:#f55">waiting for frames...</span>
          <span id="fps" style="opacity:0.6;margin-left:12px"></span>
        </div>
        <img id="feed" style="max-width:100%;border:2px solid #444;
             background:#000;min-height:200px">

        <script>
          const img    = document.getElementById('feed');
          const status = document.getElementById('status');
          const fpsEl  = document.getElementById('fps');

          let shown = 0;
          let last  = Date.now();

          // Preload the next frame, swap only when it has fully loaded.
          // This prevents the flicker you get from setting src directly.
          function tick() {
            const next = new Image();

            next.onload = () => {
              img.src = next.src;
              shown++;
              status.textContent = 'live';
              status.style.color = '#5f5';
              setTimeout(tick, 120);
            };

            next.onerror = () => {
              status.textContent = 'no frames - is the board sending?';
              status.style.color = '#f55';
              setTimeout(tick, 600);
            };

            // Date.now() busts the cache; without it the browser
            // returns the same frame forever.
            next.src = '/latest?t=' + Date.now();
          }

          setInterval(() => {
            const now = Date.now();
            fpsEl.textContent = (shown / ((now - last) / 1000)).toFixed(1) + ' fps';
            shown = 0;
            last  = now;
          }, 1000);

          tick();
        </script>

        <p style="opacity:0.5;font-size:13px">
          Updates only while the board is POSTing frames.<br>
          Board sends ~2/sec, so 2 fps here is normal and correct.
        </p>
      </body>
    </html>
    """


@app.get("/stream")
def stream():
    def gen():
        while True:
            with _lock:
                frame = _latest_jpeg
            if frame is not None:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.1)

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/health")
def health():
    return {"status": "ok"}
