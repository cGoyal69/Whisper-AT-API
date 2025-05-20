import os
import time
import tempfile
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper_at as whisper
from collections import defaultdict

# Define known vocal and instrumental tags
VOCAL_TAGS = {
    "Singing", "Speech", "Choir", "Female singing", "Male singing",
    "Chant", "Yodeling", "Shout", "Bellow", "Rapping", "Narration",
    "Child singing", "Vocal music", "Opera", "A capella", "Voice",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue", 
    "Babbling", "Speech synthesizer", "Whoop", "Yell", "Battle cry",
    "Children shouting", "Screaming", "Whispering", "Mantra",
    "Synthetic singing", "Humming", "Whistling", "Beatboxing",
    "Gospel music", "Lullaby", "Groan", "Grunt"
}

INSTRUMENTAL_TAGS = {
    "Piano", "Electric piano", "Keyboard (musical)", "Musical instrument",
    "Synthesizer", "Organ", "New-age music", "Electronic organ", 
    "Christmas music", "Rhythm and blues", "Independent music", "Soundtrack music", 
    "Pop music", "Jazz", "Soul music", "Christian music", "Instrumental music", 
    "Harpsichord", "Guitar", "Bass guitar", "Drums", "Violin", "Trumpet", 
    "Flute", "Saxophone", "Plucked string instrument", "Electric guitar", 
    "Acoustic guitar", "Steel guitar, slide guitar", "Banjo", "Sitar", "Mandolin", 
    "Ukulele", "Hammond organ", "Sampler", "Percussion", "Drum kit", 
    "Drum machine", "Drum", "Snare drum", "Bass drum", "Timpani", "Tabla", 
    "Cymbal", "Hi-hat", "Tambourine", "Marimba, xylophone", 
    "Vibraphone", "Orchestra", "Brass instrument",
    "French horn", "Trombone", "Bowed string instrument", "String section", 
    "Violin, fiddle", "Cello", "Double bass", "Wind instrument, woodwind instrument", 
    "Clarinet", "Harp", "Harmonica", "Accordion", 
    "Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock", 
    "Rock and roll", "Psychedelic rock", "Reggae", "Country", "Swing music", 
    "Bluegrass", "Funk", "Folk music", "Middle Eastern music", "Disco", 
    "Classical music", "Electronic music", "House music", "Techno", "Dubstep", 
    "Drum and bass", "Electronica", "Electronic dance music", "Ambient music", 
    "Trance music", "Music of Latin America", "Salsa music", "Flamenco", "Blues", 
    "Music for children", "Music of Africa", "Afrobeat", "Music of Asia", 
    "Carnatic music", "Music of Bollywood", "Ska", "Traditional music", 
    "Song", "Theme music", "Video game music", "Dance music", "Wedding music", 
    "Happy music", "Funny music", "Sad music", "Tender music", "Exciting music", 
    "Angry music", "Scary music"
}

app = FastAPI(
    title="Audio Classification API",
    description="API for audio classification using Whisper-AT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class AudioClassificationResponse(BaseModel):
    transcription: str
    top_tags: List[str]
    classification: str

# Cache for the model to avoid reloading
model_cache = {}

def get_model(model_size: str = "tiny"):
    """Get or load the whisper model - using tiny to save space"""
    if model_size not in model_cache:
        print(f"Loading Whisper-AT model ({model_size})...")
        start_time = time.time()
        model_cache[model_size] = whisper.load_model(model_size)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    return model_cache[model_size]

def classify_audio(top_tags):
    """Classify audio based on tags"""
    has_vocal = any(tag in VOCAL_TAGS for tag in top_tags)
    has_instrumental = any(tag in INSTRUMENTAL_TAGS for tag in top_tags)

    if has_vocal and not has_instrumental:
        return "Vocal Audio"
    elif has_instrumental and not has_vocal:
        return "Instrumental Audio"
    elif has_vocal and has_instrumental:
        return "Music"
    else:
        return "Unknown"

def process_audio_file(file_path: str, model_size: str = "tiny"):
    """Process audio file and extract classification information - simplified for disk space"""
    model = get_model(model_size)
    audio_tagging_time_resolution = 4.8
    
    # Transcribe and tag the audio
    result = model.transcribe(file_path, at_time_res=audio_tagging_time_resolution)
    
    # Parse audio tags
    audio_tag_result = whisper.parse_at_label(
        result,
        language='en',
        top_k=15,
        p_threshold=-5
    )
    
    # Process segments and collect tag frequency
    tag_freq = defaultdict(int)
    
    for segment in audio_tag_result:
        # Update tag frequency
        for tag, _ in segment['audio tags']:
            tag_freq[tag] += 1
    
    # Find top tags (those that appear more than once)
    top_tags = [tag for tag, freq in tag_freq.items() if freq > 1]
    
    # Get classification
    classification = classify_audio(top_tags)
    
    return AudioClassificationResponse(
        transcription=result["text"],
        top_tags=top_tags,
        classification=classification
    )

@app.post("/classify")
async def classify_audio_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_size: str = "tiny"
):
    """
    Upload an audio file and get classification results.
    
    - **file**: The audio file to analyze
    - **model_size**: Whisper model size (tiny, base, small)
    """
    # Validate model size - limit to smaller models to save space
    if model_size not in ["tiny", "base", "small"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model size. Choose from: tiny, base, small"
        )
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the audio
        result = process_audio_file(temp_file_path, model_size)
        
        # Clean up in the background after response is sent
        background_tasks.add_task(shutil.rmtree, temp_dir)
        
        return result
        
    except Exception as e:
        # Clean up and raise exception
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "name": "Audio Classification API",
        "version": "1.0.0",
        "description": "API for audio classification using Whisper-AT"
    }
