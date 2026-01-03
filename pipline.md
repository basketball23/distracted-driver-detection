## Inference pipeline will go:
    - Video input and resizing
    - ROI extraction (mediapipe facemesh, hand detection) -> cropping
    - Feature extraction (mobilenet)
    - Temporal smoothing using sliding window (reduce frame jitter)
    - Similarity consistency (measure similarity with previous frames, if so enforce predictions)
    - Final driver behavior output layer + confidence score
    - System output (binary distracted score, risk score, current driver state, etc)