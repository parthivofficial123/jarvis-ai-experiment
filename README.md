# JARVIS AI Experiment

This is a small experimental project where I tried to build a simple AI assistant system that combines several modern AI tools.

I built this project mainly to **learn how different AI systems interact with each other**.

I'm a high school student still learning, so this project was more about **exploration and experimentation** than building a polished assistant.

Some parts of the code were developed with help from AI tools (Anthropic's Claude) while I was learning how the pieces fit together.

---

## Features

• Speech recognition using Whisper  
• Local LLM responses using Ollama (LLaMA 3)  
• Object detection with YOLOv8  
• Webcam scene description  
• Text-to-speech responses  

---

## How It Works

1. The system records audio from the microphone.
2. Whisper transcribes the speech to text.
3. The query is sent to a local LLaMA 3 model through Ollama.
4. The response is spoken aloud using text-to-speech.
5. If asked to "look around", the system uses YOLOv8 to detect objects from the webcam.

---

## Technologies Used

Python  
OpenCV  
YOLOv8  
Whisper  
Ollama / LLaMA 3  
pyttsx3  

---

## Example Commands

"Jarvis, what do you see?"

"Jarvis, tell me something interesting."

"Stop"

---

## Notes

This project is mainly an **experiment to understand AI pipelines**.  
It's not meant to be a production-level assistant.

---

## Future Ideas

• Better wake word detection  
• More natural conversation handling  
• Integration with hardware devices  
• Smarter vision descriptions
