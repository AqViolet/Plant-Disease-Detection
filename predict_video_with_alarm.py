import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import platform
import time
from playsound import playsound  

MODEL_PATH = "trained_plant_disease_model.keras"
VALID_DIR = "data/valid"          
VIDEO_PATH = "VIDEO-2025-11-04-12-16-28.mp4"  
FRAME_SAMPLE_RATE = 5             
IMG_SIZE = (128, 128)
CONF_THRESHOLD = 0.4              
ALARM_SOUND_PATH = "alarm.wav"  

print("Loading dataset structure for class names...")
validation_set = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    shuffle=True
)
class_names = validation_set.class_names
print("Classes loaded:", class_names)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


UNHEALTHY_CLASSES = [c for c in class_names if "healthy" not in c.lower()]
print("Unhealthy classes:", UNHEALTHY_CLASSES)


def trigger_alarm(label, confidence):
    """Play a beep or custom alarm when an unhealthy plant is detected."""
    print(f"🚨 ALERT: {label} detected ({confidence:.2f}) at {time.strftime('%H:%M:%S')}")

    if ALARM_SOUND_PATH and os.path.exists(ALARM_SOUND_PATH):
        try:
            if platform.system() == "Darwin": 
                os.system(f"afplay '{ALARM_SOUND_PATH}' &")
            elif platform.system() == "Windows":
                import winsound
                winsound.PlaySound(ALARM_SOUND_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                playsound(ALARM_SOUND_PATH)
        except Exception as e:
            print(f"(Could not play custom alarm sound: {e})")
    else:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 400)
        else:
            print("\a", end="", flush=True)


def preprocess_frame(frame):
    """Resize and prepare a video frame for model input."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, IMG_SIZE)
    return np.expand_dims(frame, axis=0)

def predict_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Could not open video. Check the file path.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing video with {total_frames} frames...")

    unhealthy_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == 0:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("Sample Frame from Video")
            plt.axis("off")
            plt.show()

        if frame_idx % FRAME_SAMPLE_RATE == 0:
            input_arr = preprocess_frame(frame)
            preds = model.predict(input_arr, verbose=0)
            result_idx = np.argmax(preds)
            label = class_names[result_idx]
            conf = float(preds[0][result_idx])

            if label in UNHEALTHY_CLASSES and conf >= CONF_THRESHOLD:
                unhealthy_count += 1
                trigger_alarm(label, conf)

        frame_idx += 1

    cap.release()
    print("\n=== Video Analysis Complete ===")
    print(f"🧾 Total frames checked: {frame_idx}")
    print(f"🚨 Unhealthy detections: {unhealthy_count}")
    print("✅ Done.")

if __name__ == "__main__":
    predict_on_video(VIDEO_PATH)
