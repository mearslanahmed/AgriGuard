<div align="center">

# AgriGuard

### AI-Powered Crop Disease Detection & IoT-Based Smart Irrigation System

[![React Native](https://img.shields.io/badge/React%20Native-Expo%2054-61DAFB?logo=react&logoColor=white)](https://reactnative.dev/)
[![Node.js](https://img.shields.io/badge/Node.js-Express-339933?logo=node.js&logoColor=white)](https://nodejs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-EfficientNetB3-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Flask](https://img.shields.io/badge/Flask-HuggingFace-000000?logo=flask&logoColor=white)](https://mearslanahmed-agriguard-flask.hf.space)

[Download APK](https://github.com/mearslanahmed/AgriGuard/releases/tag/v1.0.0) · [Backend API](https://agriguard-g3hq.onrender.com) · [ML API](https://mearslanahmed-agriguard-flask.hf.space)

</div>

---

## Overview

AgriGuard is a full-stack smart farming system built for Pakistani farmers. A farmer photographs a crop leaf, the system identifies the crop and disease using a two-stage deep learning pipeline, and returns a complete pesticide advisory - dosage, spray interval, and safety guidelines. An ESP32-based hardware module handles soil moisture monitoring and automated pump control.

**10 crops · 76 disease classes · 99.15% crop accuracy · 94.86% disease accuracy**

---

## Architecture

```
Mobile App (React Native)
    │
    ├── POST /api/scans          ──▶  Express Backend (Render)
    │       └── proxy /predict   ──▶  Flask ML Server (HuggingFace)
    │
    ├── GET/POST /api/esp/*      ──▶  Express Backend
    │       └── proxy to IP      ──▶  ESP32-WROOM (local network)
    │
    └── GET /api/esp/capture     ──▶  ESP32-CAM (local network)

Express Backend ──▶ MongoDB Atlas (5 collections)
```

| Layer | Technology | Host |
|-------|-----------|------|
| Mobile | React Native / Expo 54 | GitHub Releases (APK) |
| Backend | Node.js / Express | Render |
| ML Server | Flask / TensorFlow / Keras | HuggingFace Spaces |
| Database | MongoDB Atlas | AWS Mumbai |
| Email | Brevo HTTP API | Brevo Cloud |
| Hardware | ESP32-CAM + ESP32-WROOM | Local network |

---

## ML Pipeline

Two-model architecture. A single model always predicts a class even for unsupported inputs, the crop gate solves this by rejecting unknowns before the disease classifier runs.

```
Input Image (300×300 RGB)
        │
        ▼
┌─────────────────────────────────┐
│  Model 1 — Crop Classifier      │
│  EfficientNetB3 · 11 classes    │
│  Confidence threshold: ≥ 0.45   │
└──────────────┬──────────────────┘
               │
       conf < 0.45 or Unknown?
          Yes → 422 (rejected)
          No  ↓
               │
┌─────────────────────────────────┐
│  Model 2 — Disease Classifier   │
│  EfficientNetB3 · 76 classes    │
│  Predictions filtered to crop   │
│  Confidence threshold: ≥ 0.35   │
└──────────────┬──────────────────┘
               │
       conf < 0.35?
          Yes → 422 (low confidence)
          No  → 200 {crop, disease, confidence, is_healthy}
```

### Model Performance

| Model | Classes | Test Accuracy | Macro F1 |
|-------|---------|--------------|----------|
| Crop Classifier | 11 | **99.15%** | 0.9915 |
| Disease Classifier | 76 | **94.86%** | 0.9372 |

Training: EfficientNetB3 backbone · Google Colab Pro A100 · 11 merged datasets · 300×300 input

### Supported Crops

Apple · Corn · Cotton · Grape · Mango · Potato · Rice · Sugarcane · Tomato · Wheat

---

## Features

**Disease Detection**
- Two-stage EfficientNetB3 pipeline with crop gate and disease classifier
- Runner-up crop fallback when primary disease confidence falls below threshold
- Image capture via mobile camera, gallery, or ESP32-CAM

**Pesticide Advisory**
- MongoDB-backed advisory collection covering all 76 disease classes
- Returns pesticide name, dosage, spray interval, mixing ratio, and safety precautions
- Admin-editable without code changes

**Smart Irrigation**
- ESP32-WROOM reads soil moisture via analog pin (AO), digital ADC fails with WiFi active
- Relay-controlled 5V pump with manual ON/OFF and auto mode (ON < 30%, OFF > 70%)
- Dynamic IP registration at boot, ESP resolves backend hostname once, uses IP directly to avoid DNS crashes

**Authentication**
- OTP-based email verification via Brevo HTTP API (SMTP blocked on Render free tier)
- bcrypt password hashing · JWT sessions (30-day expiry) · Expo SecureStore persistence
- Profile pictures scoped per user ID to prevent cross-account bleed

**Scan History**
- All results persisted with crop, disease label, confidence, timestamp
- User-scoped access control, users see and delete only their own scans

---

## Project Structure

```
AgriGuard/
├── backend/
│   ├── config/
│   │   ├── db.js                  # MongoDB connection
│   │   └── email.js               # Brevo HTTP API sender
│   ├── controllers/
│   │   ├── authController.js      # Auth (7 endpoints)
│   │   ├── scanController.js      # Scan CRUD
│   │   ├── pesticideController.js # Advisory lookup
│   │   └── espController.js       # IoT proxy (10 endpoints)
│   ├── models/
│   │   ├── User.js                # bcrypt pre-save hook
│   │   ├── Scan.js
│   │   ├── Pesticide.js           # disease_label indexed field
│   │   ├── OTP.js                 # TTL index (5 min)
│   │   └── EspDevice.js           # Dynamic IP registry
│   ├── middleware/authMiddleware.js
│   ├── routes/
│   ├── seeders/pesticideSeeder.js
│   └── server.js
│
├── flask_api/
│   └── app.py                     # /predict · dual-model pipeline
│
├── mobile_app/
│   └── src/
│       ├── screens/
│       ├── services/
│       ├── navigation/
│       ├── context/AuthContext.js
│       └── config.js              # BACKEND_URL · FLASK_URL
│
├── hardware/
│   ├── esp32_cam/
│   │   └── esp32_cam.ino
│   └── esp32_wroom/
│       └── pump_controller.ino
│
├── ml_model/
│   ├── AgriGuard_Model1_Final.keras    # gitignored — hosted on HuggingFace
│   ├── AgriGuard_Model2_Final.keras    # gitignored — hosted on HuggingFace
│   ├── model1_crop_mapping.json
│   ├── model2_disease_mapping.json
│   └── notebooks/
│
├── screenshots/
├── diagrams/
└── docs/
```

---

## API Reference

### Auth - `/api/auth`

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/send-otp` | — | Send OTP to email |
| POST | `/register` | — | Register with OTP verification |
| POST | `/login` | — | Login, returns JWT |
| POST | `/reset-password` | — | Reset password with OTP |
| GET | `/me` | JWT | Get current user |
| PUT | `/update` | JWT | Update profile |
| PUT | `/change-password` | JWT | Change password |

### Scans - `/api/scans`

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/` | JWT | Save scan result |
| GET | `/` | JWT | Get user's scan history |
| DELETE | `/:id` | JWT | Delete scan |

### Pesticide - `/api/pesticides`

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/:disease_label` | JWT | Get advisory by disease label |

### IoT - `/api/esp`

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/register` | HW Key | ESP device self-registration |
| GET | `/capture` | JWT | Trigger ESP32-CAM capture |
| GET | `/wroom/status` | JWT | Moisture + pump status |
| GET | `/wroom/pump/on` | JWT | Turn pump ON |
| GET | `/wroom/pump/off` | JWT | Turn pump OFF |
| GET | `/wroom/auto/on` | JWT | Enable auto irrigation |
| GET | `/wroom/auto/off` | JWT | Disable auto irrigation |

### ML - Flask `/predict`

```
POST /predict
Content-Type: multipart/form-data
Body: image (file, max 5MB)

200 → { success, crop, crop_confidence, disease, disease_label, disease_confidence, confidence, is_healthy }
422 → { success: false, error, crop, crop_confidence }
```

---

## Local Setup

### Prerequisites

- Node.js 18+
- Python 3.11.8 (required — Keras 3.13.2 incompatible with 3.12+)
- MongoDB Atlas cluster
- Expo account + EAS CLI

### Backend

```bash
cd backend
npm install
```

`.env`:
```
MONGO_URI=mongodb+srv://...
JWT_SECRET=your_secret
BREVO_API_KEY=your_brevo_key
VERIFIED_SENDER_EMAIL=your@email.com
HW_API_KEY=your_hw_secret
PORT=3000
```

```bash
node seeders/pesticideSeeder.js   # seed pesticide data
npm start
```

### Flask ML Server

```bash
cd flask_api
pip install -r requirements.txt
python app.py
```

Models are hosted on HuggingFace and loaded from the local `model/` directory when running the Space. For local development, download from [AgriGuard-Models](https://huggingface.co/mearslanahmed/AgriGuard-Models) and place in `flask_api/model/`.

### Mobile App

```bash
cd mobile_app
npm install
```

Update `src/config.js`:
```javascript
export const BACKEND_URL = 'https://agriguard-g3hq.onrender.com';
export const FLASK_URL = 'https://mearslanahmed-agriguard-flask.hf.space';
```

```bash
npx expo start          # development
eas build -p android --profile preview   # production APK
```

### ESP32 Hardware

Flash `hardware/esp32_cam/esp32_cam.ino` to ESP32-CAM (AI-Thinker).
Flash `hardware/esp32_wroom/pump_controller.ino` to ESP32-WROOM-32.

Update WiFi credentials and backend URL in both sketches before flashing.

**Wiring (WROOM):**
- Soil moisture sensor AO → GPIO 34
- Relay signal → GPIO 26
- Relay COM/NO → pump power line

**Note:** DNS resolution crashes ESP32 inside the TCP stack. Both devices resolve the backend hostname once at boot via `WiFi.hostByName()` and use the resolved IP with a `Host` header for all subsequent requests.

---

## Screenshots

<div align="center">

| Splash | Login | Register | OTP Verification |
|--------|-------|----------|-----------------|
| <img src="screenshots/Splash%20Screen.png" width="160"> | <img src="screenshots/Login%20Screen.png" width="160"> | <img src="screenshots/Registration%20Screen.png" width="160"> | <img src="screenshots/OTP%20Verification.jpg" width="160"> |

| Home | Detect | Analyzing | Result (Diseased) |
|------|--------|-----------|-------------------|
| <img src="screenshots/Home%20Screen%20(Dashboard).jpg" width="160"> | <img src="screenshots/Detect%20Screen%20(Empty).png" width="160"> | <img src="screenshots/Detect%20Screen%20(Analyzing).png" width="160"> | <img src="screenshots/Result%20Screen%20(Diseased).jpg" width="160"> |

| Result (Healthy) | History | Profile | Water Management |
|-----------------|---------|---------|-----------------|
| <img src="screenshots/Result%20Screen%20(Healthy).png" width="160"> | <img src="screenshots/History%20Screen.png" width="160"> | <img src="screenshots/Profile%20Screen.png" width="160"> | <img src="screenshots/Water%20Management.png" width="160"> |

</div>

---

## Diagrams

| Diagram | |
|---------|--|
| System Architecture | [View](diagrams/4_system_architecture.png) |
| ML Pipeline | [View](diagrams/6_ml_pipeline.png) |
| Database ERD | [View](diagrams/5_database_erd.png) |
| Use Case | [View](diagrams/1_use%20case_diagram.png) |
| Activity - Disease Detection | [View](diagrams/2_activity_disease_detection_diagram.png) |
| Sequence - Authentication | [View](diagrams/3a_sequence_authentication.png) |
| Sequence - Disease Detection | [View](diagrams/3b_sequence_disease_detection.png) |
| Sequence - Water Management | [View](diagrams/3c_sequence_water_management.png) |
| Hardware Communication | [View](diagrams/7_hardware_communication.png) |

---

## Team

Built by [Arslan Ahmed](https://github.com/mearslanahmed) (lead developer) as part of a Final Year Project at GCUF, Session 2022–2026.

| Role | Name | Contact |
|------|------|---------|
| Lead Developer | Arslan Ahmed | arslanahmednaseem@gmail.com |
| Team Member | Amna Ikram | amnaikram822@gmail.com |

---

## License

MIT License. Developed as a Final Year Project at the Department of Software Engineering, Government College University Faisalabad.

---

AgriGuard was developed for **AG Leaders** - a smart agriculture initiative. A formal project letter is on file.