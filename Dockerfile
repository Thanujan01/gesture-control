# MediaPipe's native library links against OpenGL ES at load time, even for
# headless CPU inference. Render's stock Python image lacks those .so files,
# which fails with:
#     OSError: libGLESv2.so.2: cannot open shared object file
# A native Python service cannot apt-get them, so Docker is the fix.

# Pinned to 3.11 deliberately. Render's native runtime picked Python 3.14,
# which is far ahead of what MediaPipe and scikit-learn are tested against.
FROM python:3.11-slim

# System libraries MediaPipe needs at import time:
#   libgles2      -> libGLESv2.so.2   (the error above)
#   libegl1       -> libEGL.so.1      (fails next, same cause)
#   libgl1        -> libGL.so.1       (OpenCV)
#   libglib2.0-0  -> libgthread       (OpenCV)
#   libgomp1      -> OpenMP runtime   (scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgles2 \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so pip layers cache between code-only pushes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render supplies $PORT. Shell form so the variable is expanded at runtime —
# exec form would pass the literal string "$PORT" and the service would be
# unreachable while appearing healthy.
#
# Running server_with_view rather than server: identical /detect behaviour,
# plus /view and /latest so you can SEE what the board camera is sending.
# That is the only reliable way to tell a camera problem from a model problem.
# Switch back to "server:app" once everything works, to save a little memory.
CMD uvicorn server_with_view:app --host 0.0.0.0 --port ${PORT:-8000}
