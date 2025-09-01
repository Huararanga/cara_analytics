FROM python:3.11

WORKDIR /app

# System deps for OpenCV, InsightFace, ONNXRuntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Install all deps except OpenCV first
# RUN grep -v "opencv" requirements.txt > requirements_no_cv.txt && \
#     pip install --no-cache-dir -r requirements_no_cv.txt && \
#     pip install --no-cache-dir opencv-python-headless==4.8.0.76 && \
#     python -c "import cv2; print('✅ OpenCV installed, version:', cv2.__version__)"

COPY app ./app

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
