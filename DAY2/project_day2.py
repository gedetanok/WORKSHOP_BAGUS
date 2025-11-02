import os
import sounddevice as sd
import scipy.io.wavfile as wavfile
from playsound import playsound
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

# 1. Setup client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Rekam suara dari mic dan simpan ke file .wav
def record_voice(filename="user.wav", duration=5, sample_rate=44100):
    print("Silakan bicara (5 detik)...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    wavfile.write(filename, sample_rate, audio)
    print("Rekaman selesai.\n")
    return filename

# 3. Speech to Text
def speech_to_text(audio_path):
    with open(audio_path, "rb") as f:
        text = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    print("Kamu bilang:", text, "\n")
    return str(text)

# 4. Chat dengan AI (pakai history)
def ask_ai(user_text, chat_history):
    # Tambahkan pesan baru user ke history
    chat_history.append({"role": "user", "content": user_text})

    # Kirim seluruh riwayat percakapan ke model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_history,
        temperature=0.7,
    )

    ai_answer = response.choices[0].message.content
    print("AI:", ai_answer, "\n")

    # Tambahkan balasan AI ke history
    chat_history.append({"role": "assistant", "content": ai_answer})
    return ai_answer, chat_history

# 5. Text to Speech
def text_to_speech(text, out_file="assistant.mp3"):
    out_path = Path(out_file)
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    ) as resp:
        resp.stream_to_file(out_path)
    print("Memutar suara AI...\n")
    playsound(str(out_path))

# 6. Main loop dengan history
def main():
    print("Voice Chat AI")
    print("Ketik 'exit' kapan saja untuk keluar.\n")

    chat_history = [
        {"role": "system", "content": "Kamu adalah asisten AI yang ramah, jawab singkat dan santai dalam bahasa Indonesia."}
    ]  # inisialisasi history kosong

    while True:
        audio_path = record_voice(duration=5)
        user_text = speech_to_text(audio_path).strip().lower()

        if "exit" in user_text or "keluar" in user_text:
            print("Sampai jumpa!")
            break

        ai_text, chat_history = ask_ai(user_text, chat_history)
        text_to_speech(ai_text)

if __name__ == "__main__":
    main()
