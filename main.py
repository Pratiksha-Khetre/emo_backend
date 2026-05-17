from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
import base64
import requests
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="EmoTune API", version="2.0.0")

# ========== CORS CONFIGURATION ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== CONFIGURATION ==========
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "./models/raf_db_model.h5")
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_HEIGHT = 75
IMG_WIDTH = 75

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

model = None
model_loading_error = None

# ========== EMOTION & LANGUAGE SEARCH QUERIES ==========
# Each emotion+language combo maps to a YouTube search query
EMOTION_LANGUAGE_QUERIES = {
    "Happy": {
        "Hindi":    "happy bollywood songs hindi hits",
        "English":  "happy pop songs upbeat feel good",
        "Marathi":  "happy marathi songs आनंदी",
        "Telugu":   "happy telugu songs dance hits",
        "Tamil":    "happy tamil songs kuthu",
        "Gujarati": "happy garba gujarati songs",
        "Urdu":     "happy urdu songs romantic",
        "Kannada":  "happy kannada songs dance",
        "Bengali":  "happy bengali songs আনন্দ",
        "Malayalam":"happy malayalam songs dance",
    },
    "Sad": {
        "Hindi":    "sad bollywood songs dard hindi",
        "English":  "sad songs emotional heartbreak",
        "Marathi":  "sad marathi songs दुःख",
        "Telugu":   "sad telugu songs emotional",
        "Tamil":    "sad tamil songs kadhal",
        "Gujarati": "sad gujarati songs bhajan",
        "Urdu":     "ghazal sad urdu poetry songs",
        "Kannada":  "sad kannada songs emotional",
        "Bengali":  "sad bengali songs রবীন্দ্রনাথ",
        "Malayalam":"sad malayalam songs emotional",
    },
    "Angry": {
        "Hindi":    "angry hindi rap songs aggressive",
        "English":  "angry rock metal intense songs",
        "Marathi":  "powerful marathi songs energy",
        "Telugu":   "mass telugu songs powerful",
        "Tamil":    "mass tamil songs powerful beats",
        "Gujarati": "energetic gujarati dandiya",
        "Urdu":     "powerful urdu rap songs",
        "Kannada":  "mass kannada songs powerful",
        "Bengali":  "powerful bengali songs rock",
        "Malayalam":"powerful malayalam songs mass",
    },
    "Fear": {
        "Hindi":    "dark hindi songs mysterious",
        "English":  "dark ambient mysterious songs",
        "Marathi":  "dark marathi songs mysterious",
        "Telugu":   "dark telugu songs thriller",
        "Tamil":    "dark tamil songs thriller bgm",
        "Gujarati": "dark gujarati devotional songs",
        "Urdu":     "dark urdu sufi mysterious",
        "Kannada":  "dark kannada songs thriller",
        "Bengali":  "dark bengali songs mysterious",
        "Malayalam":"dark malayalam songs thriller",
    },
    "Disgust": {
        "Hindi":    "alternative hindi indie songs",
        "English":  "alternative indie punk rock songs",
        "Marathi":  "indie marathi songs alternative",
        "Telugu":   "indie telugu songs alternative",
        "Tamil":    "indie tamil songs alternative",
        "Gujarati": "indie gujarati folk songs",
        "Urdu":     "indie urdu songs alternative",
        "Kannada":  "indie kannada songs alternative",
        "Bengali":  "indie bengali band songs",
        "Malayalam":"indie malayalam songs alternative",
    },
    "Surprise": {
        "Hindi":    "energetic bollywood dance party songs",
        "English":  "upbeat exciting dance pop songs",
        "Marathi":  "energetic marathi lavani dance",
        "Telugu":   "energetic telugu dance party songs",
        "Tamil":    "energetic tamil dance party songs",
        "Gujarati": "energetic gujarati garba dandiya",
        "Urdu":     "energetic urdu party songs",
        "Kannada":  "energetic kannada dance songs",
        "Bengali":  "energetic bengali dance songs",
        "Malayalam":"energetic malayalam dance songs",
    },
    "Neutral": {
        "Hindi":    "chill bollywood lofi hindi songs",
        "English":  "chill lofi relaxing ambient songs",
        "Marathi":  "chill marathi songs peaceful",
        "Telugu":   "chill telugu lofi songs",
        "Tamil":    "chill tamil lofi songs",
        "Gujarati": "chill gujarati songs peaceful",
        "Urdu":     "chill sufi urdu songs peaceful",
        "Kannada":  "chill kannada songs peaceful",
        "Bengali":  "chill bengali rabindra sangeet",
        "Malayalam":"chill malayalam songs peaceful",
    },
}

# ========== LOAD MODEL ==========
def load_model_safe():
    global model, model_loading_error

    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} loaded")
    except ImportError as e:
        model_loading_error = "TensorFlow not installed"
        print(f"✗ TensorFlow import failed: {e}")
        return False

    if not os.path.exists(MODEL_PATH):
        model_loading_error = f"Model file not found at {MODEL_PATH}"
        print(f"✗ {model_loading_error}")
        return False

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
        model_loading_error = None
        return True

    except TypeError as e:
        print(f"⚠️  Standard load failed ({e}), trying custom_object_scope fallback...")
        try:
            import tensorflow as tf
            from tensorflow.keras.layers import InputLayer
            from tensorflow.keras.utils import custom_object_scope
            with custom_object_scope({'InputLayer': InputLayer}):
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✓ Model loaded via custom_object_scope")
            model_loading_error = None
            return True
        except Exception as e2:
            model_loading_error = str(e2)
            print(f"✗ Fallback also failed: {e2}")
            return False

    except Exception as e:
        model_loading_error = str(e)
        print(f"✗ Error loading model: {e}")
        return False

# ========== STARTUP ==========
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*50)
    print("🚀 EmoTune API v2.0 Starting (YouTube Edition)...")
    print("="*50)
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {os.sys.version}")
    print(f"📦 PORT: {os.getenv('PORT', '8000')}")
    print(f"🎬 YouTube API configured: {bool(YOUTUBE_API_KEY)}")

    print("\n📦 Loading model...")
    model_ok = load_model_safe()

    if model_ok:
        print("✅ Model loaded - emotion detection available")
    else:
        print("⚠️  Model not loaded - emotion detection unavailable")
        print(f"   Error: {model_loading_error}")

    print("\n" + "="*50)
    print("✅ Server is ready!")
    print("="*50 + "\n")

# ========== HEALTH CHECK ==========
@app.get("/")
async def root():
    return {
        "message": "EmoTune API",
        "status": "running",
        "version": "2.0.0",
        "youtube_configured": bool(YOUTUBE_API_KEY),
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_error": model_loading_error,
        "youtube_configured": bool(YOUTUBE_API_KEY),
        "features": {
            "emotion_detection": model is not None,
            "music_recommendations": bool(YOUTUBE_API_KEY),
        }
    }

# ========== YOUTUBE SEARCH ==========
def search_youtube_tracks(emotion: str, language: str, max_results: int = 10):
    """Search YouTube for music matching emotion and language"""
    if not YOUTUBE_API_KEY:
        print("⚠️  No YouTube API key configured")
        return []

    emotion = emotion if emotion in EMOTION_LANGUAGE_QUERIES else "Neutral"
    language = language if language in EMOTION_LANGUAGE_QUERIES.get(emotion, {}) else "English"

    query = EMOTION_LANGUAGE_QUERIES[emotion][language]
    print(f"🎬 YouTube search: '{query}'")

    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "videoCategoryId": "10",  # Music category
                "maxResults": max_results,
                "key": YOUTUBE_API_KEY,
                "safeSearch": "moderate",
                "relevanceLanguage": "en",
            },
            timeout=10
        )

        if response.status_code != 200:
            print(f"YouTube API error: {response.status_code} {response.text}")
            return []

        items = response.json().get("items", [])
        tracks = []

        for item in items:
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if not video_id:
                continue

            thumbnail = (
                snippet.get("thumbnails", {})
                .get("high", snippet.get("thumbnails", {}).get("default", {}))
                .get("url", "")
            )

            tracks.append({
                "id": video_id,
                "title": snippet.get("title", "Unknown"),
                "artist": snippet.get("channelTitle", "Unknown"),
                "image_url": thumbnail,
                "external_url": f"https://www.youtube.com/watch?v={video_id}",
                "embed_url": f"https://www.youtube.com/embed/{video_id}",
                "language": language,
                "emotion": emotion,
                "source": "youtube",
            })

        print(f"✓ Found {len(tracks)} YouTube tracks")
        return tracks

    except Exception as e:
        print(f"✗ YouTube search error: {e}")
        return []

# ========== RECOMMENDATIONS ENDPOINT ==========
@app.get("/get_recommendations/")
async def get_recommendations(
    emotion: str = Query("Neutral"),
    languages: str = Query("English"),
    offset: int = Query(0)
):
    language_list = [lang.strip() for lang in languages.split(",") if lang.strip()]
    if not language_list:
        language_list = ["English"]

    print(f"\n📍 Request: emotion={emotion}, languages={language_list}, offset={offset}")

    all_recommendations = []

    for language in language_list:
        results = search_youtube_tracks(emotion, language, max_results=10)
        for track in results:
            if not any(t["id"] == track["id"] for t in all_recommendations):
                all_recommendations.append(track)

    total_available = len(all_recommendations)
    paginated = all_recommendations[offset:offset + 10]

    return {
        "recommendations": paginated,
        "emotion": emotion,
        "languages": language_list,
        "total_available": total_available,
        "offset": offset,
        "returned_count": len(paginated),
        "source": "youtube",
    }

# ========== EMOTION DETECTION ==========
def process_and_predict(image_file_bytes):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not available",
                "message": model_loading_error or "Model not loaded",
            }
        )

    nparr = np.frombuffer(image_file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected in image")

    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
    face_roi = img_gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))

    processed_input = face_resized.astype('float32') / 255.0
    processed_input = np.expand_dims(processed_input, axis=-1)
    processed_input = np.expand_dims(processed_input, axis=0)

    predictions = model.predict(processed_input, verbose=0)
    probabilities = predictions[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_emotion = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    text = f"{predicted_emotion} ({confidence*100:.1f}%)"
    cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpeg', img_bgr)
    processed_image_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer.tobytes()).decode('utf-8')

    return {
        "processed_image_b64": processed_image_b64,
        "predicted_emotion": predicted_emotion,
        "confidence": confidence,
        "all_confidences": {
            CLASS_NAMES[i]: round(float(probabilities[i]), 4)
            for i in range(len(CLASS_NAMES))
        }
    }

@app.post("/analyze_emotion/")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = process_and_predict(image_bytes)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== RUN ==========
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
