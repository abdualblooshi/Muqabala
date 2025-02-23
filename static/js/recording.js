class AudioRecorder {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.stream = null;
  }

  async startRecording() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(this.stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        this.audioChunks.push(event.data);
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      return true;
    } catch (error) {
      console.error("Error accessing microphone:", error);
      return false;
    }
  }

  stopRecording() {
    return new Promise((resolve) => {
      if (!this.mediaRecorder) {
        resolve(null);
        return;
      }

      this.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(this.audioChunks, { type: "audio/wav" });
        const base64data = await this.blobToBase64(audioBlob);
        this.cleanup();
        resolve(base64data);
      };

      this.mediaRecorder.stop();
      this.isRecording = false;
    });
  }

  cleanup() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
    }
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.stream = null;
  }

  blobToBase64(blob) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.readAsDataURL(blob);
    });
  }
}

class RecordingUI {
  constructor() {
    this.recorder = new AudioRecorder();
    this.button = document.getElementById("recordButton");
    this.setupEventListeners();
  }

  setupEventListeners() {
    if (!this.button) return;

    this.button.addEventListener("click", async () => {
      if (!this.recorder.isRecording) {
        const success = await this.recorder.startRecording();
        if (success) {
          this.updateButtonState(true);
        } else {
          alert(
            "Error accessing microphone. Please check permissions and try again."
          );
        }
      } else {
        this.updateButtonState(false);
        const audioData = await this.recorder.stopRecording();
        if (audioData) {
          await this.processAudio(audioData);
        }
      }
    });
  }

  updateButtonState(isRecording) {
    if (!this.button) return;

    if (isRecording) {
      this.button.classList.add("recording");
      this.button.querySelector("span:last-child").textContent = "🎤 Stop";
    } else {
      this.button.classList.remove("recording");
      this.button.querySelector("span:last-child").textContent = "🎤 Record";
    }
  }

  async processAudio(audioData) {
    const inputElement = document.querySelector(
      'input[aria-label="Response Input"]'
    );
    if (!inputElement) return;

    try {
      inputElement.value = "Transcribing...";
      inputElement.dispatchEvent(new Event("input", { bubbles: true }));

      const response = await fetch("/process_audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio: audioData }),
      });

      if (!response.ok) throw new Error("Network response was not ok");
      const data = await response.json();

      inputElement.value = data.text;
      inputElement.dispatchEvent(new Event("input", { bubbles: true }));
    } catch (error) {
      console.error("Error processing audio:", error);
      inputElement.value = "Error transcribing audio. Please try again.";
      inputElement.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }
}

// Initialize recording functionality when the DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new RecordingUI();
});
