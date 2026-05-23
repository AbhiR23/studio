# Skin Lesion Analysis AI Platform

A comprehensive, full-stack artificial intelligence platform designed to analyze, classify, and explain skin lesion images. This project combines a modern Next.js frontend, a robust FastAPI backend for deep learning inference, and Google Genkit for conversational AI capabilities.

## 🚀 Architecture Overview

This project is divided into three core components:

### 1. Frontend (Next.js Web Application)
Located in `src/`, this is a responsive, modern web interface built with **Next.js 15** and **React 19**.
- **UI Framework:** Styled using **Tailwind CSS** and heavily utilizes **Radix UI / shadcn** primitives for a premium, accessible user experience.
- **Authentication & Database:** Integrates tightly with **Firebase** (Auth, Firestore, and Storage) to manage user sessions, store historical predictions, and securely handle uploaded lesion images.
- **AI Chatbot:** A dedicated conversational interface (`src/ai` and `src/app/chatbot`) powered by **Google Genkit AI** (`@genkit-ai/google-genai`). It allows users to ask follow-up questions about skin health and model predictions.
- **Routes:** 
  - `/dashboard`: The main hub for uploading images and viewing analysis history.
  - `/auth`: Secure Firebase-backed login/signup.
  - `/chatbot`: An interactive AI assistant.
  - `/info`: Educational resources regarding skin conditions.

### 2. Backend (FastAPI Inference Server)
Located in `backend/app.py`, this server handles the heavy lifting of Computer Vision and Deep Learning.
- **Model:** Uses a PyTorch **EfficientNet-B3** model trained to classify skin lesions into 9 distinct categories (Melanoma, Nevus, Basal Cell Carcinoma, etc.).
- **Image Quality Guardrails:** Before the model even sees an image, the backend uses **OpenCV** to run extensive heuristics:
  - Checks brightness, contrast, and blur (Laplacian variance).
  - Uses color-space conversions (YCrCb and HSV) to ensure the image actually contains human skin.
  - Reject poor-quality images to prevent the AI from making blind guesses ("Garbage in, garbage out").
- **Explainable AI (XAI) / Grad-CAM:** The `/segment` endpoint generates a heatmap overlay showing exactly *where* the model is looking to make its prediction. It highlights the primary lesion region, calculates the bounding box, and even estimates areas of potential necrosis based on saturation and brightness thresholds.

### 3. Experimental OOD Pipeline
Located in the root directory (`ood_pipeline.py`), this is an experimental 2-stage pipeline designed to catch "Out-Of-Domain" (OOD) images (e.g., pictures of cats or blurry landscapes) before classification.
- **Stage 1 (OOD Detector):** A modified PyTorch ResNet18 model trained to distinguish skin lesions from random COCO dataset images.
- **Stage 2 (7-Class Classifier):** If the image is confidently flagged as a lesion, it is passed to a secondary classifier.
- Includes a confidence thresholding logic system that outputs "Uncertain" if the model's confidence is too low.

---

## 🛠️ Tech Stack

**Frontend:** Next.js 15, React 19, Tailwind CSS, TypeScript, Firebase, Genkit AI, React Hook Form, Zod.
**Backend:** FastAPI, Python 3.12, PyTorch, OpenCV, Uvicorn, PIL.
**Data Science:** EfficientNet-B3, ResNet18, Grad-CAM (Explainable AI).

---

## ⚙️ Running the Project Locally

### Prerequisites
- Node.js (v20+)
- Python 3.12
- Firebase CLI (for emulators)

### 1. Start the Frontend
Navigate to the root folder:
```bash
npm install
npm run dev
```
*This will start the Next.js server on port 9002.*

### 2. Start the Firebase Emulators (Optional but Recommended)
For local database/auth testing:
```bash
npm run emu
```

### 3. Start the FastAPI Backend
Open a new terminal, navigate to the root folder, and use the dedicated Python 3.12 virtual environment:
```bash
# Activate the environment (Windows)
.\venv_gpu\Scripts\activate

# Install requirements (if not already installed)
pip install -r backend/requirements.txt fastapi uvicorn opencv-python pydantic

# Run the server
cd backend
python app.py
```
*The FastAPI server will be available at `http://localhost:8000`.*

### 4. Run the Genkit AI Chatbot
In a separate terminal, start the Genkit development server:
```bash
npm run genkit:dev
```

---

## ⚠️ Medical Disclaimer
This application is designed for research, educational, and demonstrative purposes only. The deep learning models (EfficientNet / ResNet) are not a substitute for professional medical diagnosis. Always consult a certified dermatologist for medical advice and diagnosis of skin conditions.
