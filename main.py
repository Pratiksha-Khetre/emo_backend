from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
import base64
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="EmoTune API", version="1.0.0")

# ========== CORS CONFIGURATION ==========
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://emotune-nine.vercel.app,https://*.vercel.app,http://localhost:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ALLOWED_ORIGINS in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== CONFIGURATION ==========
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "./models/raf_db_model.h5")
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_HEIGHT = 75
IMG_WIDTH = 75

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

spotify_tokens = {}
model = None
model_loading_error = None

# ========== EMOTION & LANGUAGE KEYWORDS ==========
EMOTION_KEYWORDS = {
    "Happy": ["happy", "cheerful", "uplifting", "feel good", "positive"],
    "Sad": ["sad", "melancholic", "emotional", "soulful", "deep"],
    "Angry": ["angry", "aggressive", "intense", "powerful", "rock"],
    "Fear": ["scary", "horror", "dark", "ominous", "creepy"],
    "Disgust": ["edgy", "dark", "alternative", "rebellious", "punk"],
    "Surprise": ["energetic", "exciting", "upbeat", "dance", "fun"],
    "Neutral": ["relaxing", "calm", "ambient", "peaceful", "chill"],
}

LANGUAGE_KEYWORDS = {
    "Hindi": ["hindi", "bollywood", "indian", "desi", "hindi songs"],
    "English": ["english", "pop", "rock", "rap", "american", "british"],
    "Marathi": ["marathi", "maharashtra", "marathi songs", "marathi music"],
    "Telugu": ["telugu", "telangana", "telugu songs", "telugu music"],
    "Tamil": ["tamil", "tamilnadu", "tamil songs", "tamil music"],
    "Gujarati": ["gujarati", "gujarati songs", "gujarati music", "gujarati folk"],
    "Urdu": ["urdu", "ghazal", "urdu poetry", "sufi", "qawwali"],
    "Kannada": ["kannada", "karnataka", "kannada songs", "kannada music"],
    "Bengali": ["bengali", "bengal", "bengali songs", "bengali folk"],
    "Malayalam": ["malayalam", "kerala", "malayalam songs", "malayalam music"],
}

# ========== LOAD MODEL (ASYNC) ==========
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
        # Primary attempt
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
        model_loading_error = None
        return True

    except TypeError as e:
        # Keras version mismatch — try rebuilding from weights only
        print(f"⚠️  Standard load failed ({e}), trying custom_object_scope fallback...")
        try:
            from tensorflow.keras.layers import InputLayer
            from tensorflow.keras.utils import custom_object_scope

            with custom_object_scope({'InputLayer': InputLayer}):
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"✓ Model loaded via custom_object_scope")
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

# ========== STARTUP EVENT ==========
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("\n" + "="*50)
    print("🚀 EmoTune API Starting...")
    print("="*50)
    
    # Print environment info
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {os.sys.version}")
    print(f"📦 PORT: {os.getenv('PORT', '8000')}")
    
    # Check Spotify config
    spotify_ok = bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)
    print(f"🎵 Spotify configured: {spotify_ok}")
    
    if SPOTIFY_REDIRECT_URI:
        print(f"   Redirect URI: {SPOTIFY_REDIRECT_URI}")
    
    # Try to load model (non-blocking)
    print("\n📦 Loading model...")
    model_ok = load_model_safe()
    
    if model_ok:
        print("✅ Model loaded - emotion detection available")
    else:
        print("⚠️  Model not loaded - emotion detection unavailable")
        print(f"   Error: {model_loading_error}")
        print("   ℹ️  Music recommendations will still work!")
    
    print("\n" + "="*50)
    print("✅ Server is ready!")
    print("="*50 + "\n")

# ========== HEALTH CHECK (ALWAYS WORKS) ==========
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "EmoTune API",
        "status": "running",
        "version": "1.0.0",
    }

@app.get("/health")
async def health_check():
    """Health check - always returns 200 even if model not loaded"""
    return {
        "status": "healthy",
        "service": "emotune-api",
        "model_loaded": model is not None,
        "model_error": model_loading_error,
        "spotify_configured": bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET),
        "features": {
            "emotion_detection": model is not None,
            "music_recommendations": True,
            "spotify_integration": bool(SPOTIFY_CLIENT_ID)
        }
    }

# ========== SPOTIFY FUNCTIONS ==========

@app.get("/spotify/login")
async def spotify_login():
    """Generate Spotify OAuth URL"""
    if not SPOTIFY_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Spotify not configured")
    
    if not SPOTIFY_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="SPOTIFY_REDIRECT_URI not set")
    
    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": "streaming user-read-private user-read-email",
    }
    auth_url = f"https://accounts.spotify.com/authorize?{urlencode(params)}"
    return {"auth_url": auth_url}

@app.get("/spotify/callback")
async def spotify_callback(code: str = Query(None), error: str = Query(None)):
    """Handle Spotify OAuth callback"""
    if error:
        return HTMLResponse(
            "<html><body style='background: #0f0f1c; color: #f0f0f0; font-family: Arial; padding: 50px; text-align: center;'>"
            "<h2>Error connecting to Spotify</h2><script>window.close();</script></body></html>"
        )
    
    if not code:
        return HTMLResponse(
            "<html><body style='background: #0f0f1c; color: #f0f0f0; font-family: Arial; padding: 50px; text-align: center;'>"
            "<h2>No authorization code received</h2></body></html>"
        )
    
    try:
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET,
        }
        
        response = requests.post("https://accounts.spotify.com/api/token", data=token_data, timeout=10)
        
        if response.status_code != 200:
            print(f"Spotify token exchange failed: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Token exchange failed")
        
        tokens = response.json()
        spotify_tokens["current_user"] = tokens
        
        return HTMLResponse(
            "<html><body style='background: #0f0f1c; color: #f0f0f0; font-family: Arial; padding: 50px; text-align: center;'>"
            "<h2>✓ Connected to Spotify!</h2><p>You can close this window.</p>"
            "<script>setTimeout(() => window.close(), 1500);</script></body></html>"
        )
    except Exception as e:
        print(f"Error in spotify_callback: {e}")
        return HTMLResponse(
            f"<html><body style='background: #0f0f1c; color: #f0f0f0; font-family: Arial; padding: 50px; text-align: center;'>"
            f"<h2>Error: {str(e)}</h2></body></html>"
        )

def get_spotify_token():
    """Get valid Spotify access token"""
    try:
        # Try user token first
        if "current_user" in spotify_tokens:
            return spotify_tokens["current_user"].get("access_token")
        
        # Fall back to client credentials
        if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
            return None
        
        auth_str = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
        auth_bytes = base64.b64encode(auth_str.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {auth_bytes}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}
        
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            headers=headers,
            data=data,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            print(f"Spotify auth failed: {response.status_code} {response.text}")
            return None
            
    except Exception as e:
        print(f"Error getting Spotify token: {e}")
        return None

def search_spotify_tracks_by_emotion_and_language(emotion, language):
    """Search Spotify for tracks matching emotion and language"""
    try:
        token = get_spotify_token()
        if not token:
            print("Could not get Spotify token")
            return []
        
        emotion = emotion if emotion in EMOTION_KEYWORDS else "Neutral"
        language = language if language in LANGUAGE_KEYWORDS else "English"
        
        emotion_keywords = EMOTION_KEYWORDS[emotion]
        language_keywords = LANGUAGE_KEYWORDS[language]
        all_tracks = []
        
        print(f"\n🎵 Searching: {emotion} + {language}")
        
        # Search with combined keywords
        for emotion_kw in emotion_keywords[:3]:
            for lang_kw in language_keywords[:2]:
                try:
                    combined_query = f"{emotion_kw} {lang_kw}"
                    
                    response = requests.get(
                        "https://api.spotify.com/v1/search",
                        headers={"Authorization": f"Bearer {token}"},
                        params={
                            "q": combined_query,
                            "type": "track",
                            "limit": 15,
                            "market": "US"
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        tracks = results.get("tracks", {}).get("items", [])
                        
                        for track in tracks:
                            if not any(t["id"] == track["id"] for t in all_tracks):
                                album_image = ""
                                if track.get("album", {}).get("images"):
                                    album_image = track["album"]["images"][0]["url"]
                                
                                track_data = {
                                    "id": track["id"],
                                    "title": track["name"],
                                    "artist": ", ".join([a["name"] for a in track["artists"]]),
                                    "image_url": album_image,
                                    "external_url": track.get("external_urls", {}).get("spotify", ""),
                                    "embed_url": f"https://open.spotify.com/embed/track/{track['id']}",
                                }
                                all_tracks.append(track_data)
                                
                except Exception as e:
                    print(f"Search error for '{combined_query}': {e}")
                    continue
        
        print(f"Found {len(all_tracks)} tracks")
        return all_tracks[:20]
        
    except Exception as e:
        print(f"Error in search_spotify_tracks: {e}")
        return []

# ========== EMOTION DETECTION ==========

def process_and_predict(image_file_bytes):
    """Process image and predict emotion"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not available",
                "message": model_loading_error or "Model not loaded",
                "suggestion": "Please contact administrator or check deployment logs"
            }
        )
    
    # Decode image
    nparr = np.frombuffer(image_file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected in image")
    
    # Use largest face
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Extract and preprocess
    face_roi = img_gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))
    
    processed_input = face_resized.astype('float32') / 255.0
    processed_input = np.expand_dims(processed_input, axis=-1)
    processed_input = np.expand_dims(processed_input, axis=0)
    
    # Predict
    predictions = model.predict(processed_input, verbose=0)
    probabilities = predictions[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_emotion = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])
    
    # Annotate image
    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    text = f"{predicted_emotion} ({confidence*100:.1f}%)"
    cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Encode to base64
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
    """Analyze emotion from uploaded image"""
    try:
        image_bytes = await file.read()
        result = process_and_predict(image_bytes)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_recommendations/")
async def get_recommendations(
    emotion: str = Query("Neutral"),
    languages: str = Query("English"),
    offset: int = Query(0)
):
    """Get song recommendations based on emotion and languages"""
    
    # Parse languages
    language_list = [lang.strip() for lang in languages.split(",") if lang.strip()]
    if not language_list:
        language_list = ["English"]
    
    print(f"\n📍 Request: emotion={emotion}, languages={language_list}, offset={offset}")
    
    all_recommendations = []
    
    # Search for each language
    for language in language_list:
        language_results = search_spotify_tracks_by_emotion_and_language(emotion, language)
        for track in language_results:
            track["language"] = language
            if not any(t["id"] == track["id"] for t in all_recommendations):
                all_recommendations.append(track)
    
    # Paginate results
    total_available = len(all_recommendations)
    paginated_recommendations = all_recommendations[offset:offset + 20]
    
    return {
        "recommendations": paginated_recommendations,
        "emotion": emotion,
        "languages": language_list,
        "total_available": total_available,
        "offset": offset,
        "returned_count": len(paginated_recommendations)
    }

# ========== RUN ==========
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
