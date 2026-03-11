import whisper
import sounddevice as sd
import scipy.io.wavfile
import numpy as np
import pyttsx3
import time
import requests
import cv2
import torch
from ultralytics import YOLO  # YOLOv8

# === Init ===
model = whisper.load_model("small")
engine = pyttsx3.init()
engine.setProperty('rate', 180)

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8n.pt")  # Will auto-download if not present

# === Audio Settings ===
SAMPLE_RATE = 16000
DURATION = 5  # seconds

def speak(text):
    print(f"JARVIS: {text}")
    engine.say(text)
    engine.runAndWait()

def record_audio(filename="temp.wav", duration=DURATION):
    print("🎙️ Listening...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    scipy.io.wavfile.write(filename, SAMPLE_RATE, recording)
    return filename

def transcribe(filename):
    print("🧠 Transcribing with Whisper...")
    result = model.transcribe(filename)
    text = result["text"].strip()
    print(f"You said: {text}")
    return text.lower() if text else None

def ask_llama(prompt):
    try:
        print("📡 Sending prompt to LLaMA 3...")
        funny_prompt = f"Give a short, funny answer to this, but if it's interesting, feel free to elaborate a bit: {prompt}"
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": funny_prompt, "stream": False}
        )
        print("🌐 RAW:", response.text)
        result = response.json()
        reply = result.get("response", "").strip()
        return reply if reply else "I'm not sure how to respond."
    except Exception as e:
        print("⚠️ Error talking to LLaMA 3:", e)
        return "There was a problem with the local model."

def capture_frame():
    print("📸 Capturing webcam frame...")
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    if not ret:
        print("⚠️ Failed to capture image.")
        return None
    return frame

def describe_scene(frame):
    print("🧠 Using YOLOv8n to detect objects...")
    results = yolo_model.predict(frame, conf=0.5, verbose=False)[0]
    names = yolo_model.names

    detected = []
    for cls in results.boxes.cls:
        name = names[int(cls)]
        if name not in detected:
            detected.append(name)

    if not detected:
        return "I can't really see anything I recognize."

    objects = ", ".join(detected)
    return f"I see the following: {objects}."

# === MAIN LOOP ===
if __name__ == "__main__":
    speak("Hello Parthiv. JARVIS with real-time YOLO vision is online!")

    while True:
        filename = record_audio(duration=5)
        query = transcribe(filename)

        if not query:
            speak("Sorry, I didn't catch that. Either you whispered, or I need better ears.")
            continue

        if "stop" in query or "exit" in query:
            speak("Peace out, Parthiv. JARVIS signing off.")
            break

        elif "what do you see" in query or "look around" in query:
            frame = capture_frame()
            if frame is not None:
                description = describe_scene(frame)
                speak(description)
            else:
                speak("Sorry, I couldn't access the webcam.")
            continue

        reply = ask_llama(query)
        speak(reply)

        time.sleep(0.5)
