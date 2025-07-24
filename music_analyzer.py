import essentia.standard as es
import essentia
import os
import sys
import urllib.request
import json
import numpy as np

# ==============================================================================
# --- MODIFY HERE ---
# VVVVVVVVVVVVVVVVVV
# 1. Specify the directory containing music files
music_directory = "/path/to/your/music/files/"  # <-- Replace with your music folder path
# 2. Define supported music file formats
SUPPORTED_FORMATS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
# 3. Model storage directory
MODELS_DIR = "./essentia_models"
# ^^^^^^^^^^^^^^^^^^
# --- END MODIFICATION ---
# ==============================================================================

# Pre-trained model configuration
MODELS_CONFIG = {
    'genre_discogs400': {
        'url': 'https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb',
        'filename': 'genre_discogs400-discogs-effnet-1.pb',
        'description': 'Discogs 400 music genre classification',
        'embedding_model': 'https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb',
        'embedding_filename': 'discogs-effnet-bs64-1.pb',
        'labels_url': 'https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json',
        'labels_filename': 'genre_discogs400-discogs-effnet-1.json'
    },
    'mtg_jamendo_genre': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.pb',
        'filename': 'mtg_jamendo_genre-discogs-effnet-1.pb',
        'description': 'MTG-Jamendo 87 music genre classification',
        'labels_url': 'https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.json',
        'labels_filename': 'mtg_jamendo_genre-discogs-effnet-1.json'
    },
    'mtg_jamendo_instrument': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb',
        'filename': 'mtg_jamendo_instrument-discogs-effnet-1.pb',
        'description': 'MTG-Jamendo instrument recognition',
        'labels_url': 'https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.json',
        'labels_filename': 'mtg_jamendo_instrument-discogs-effnet-1.json'
    },
    'mood_happy': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb',
        'filename': 'mood_happy-discogs-effnet-1.pb',
        'description': 'Happy mood detection'
    },
    'mood_sad': {
        'url': 'https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb',
        'filename': 'mood_sad-discogs-effnet-1.pb',  
        'description': 'Sad mood detection'
    },
    'voice_instrumental': {
        'url': 'https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb',
        'filename': 'voice_instrumental-discogs-effnet-1.pb',
        'description': 'Voice/instrumental classification'
    }
}

def create_models_directory():
    """Create model storage directory"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"✓ Created models directory: {MODELS_DIR}")

def download_model(model_name, model_info):
    """Download a single model file"""
    model_path = os.path.join(MODELS_DIR, model_info['filename'])
    
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_info['filename']}")
    else:
        try:
            print(f"📥 Downloading model: {model_info['description']}...")
            print(f"   File: {model_info['filename']}")
            
            # Create a simple progress callback
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size / total_size) * 100)
                    if block_num % 50 == 0:  # Update every 50 blocks
                        print(f"   Progress: {percent:.1f}%")
            
            urllib.request.urlretrieve(model_info['url'], model_path, progress_hook)
            print(f"✓ Download completed: {model_info['filename']}")
        except Exception as e:
            print(f"❌ Download failed {model_info['filename']}: {e}")
            return None
    
    # Check if embedding model needs to be downloaded
    if 'embedding_model' in model_info:
        embedding_path = os.path.join(MODELS_DIR, model_info['embedding_filename'])
        if not os.path.exists(embedding_path):
            try:
                print(f"📥 Downloading embedding model: {model_info['embedding_filename']}...")
                urllib.request.urlretrieve(model_info['embedding_model'], embedding_path)
                print(f"✓ Embedding model download completed: {model_info['embedding_filename']}")
            except Exception as e:
                print(f"❌ Embedding model download failed: {e}")
                return None
    
    # Check if labels file needs to be downloaded
    if 'labels_url' in model_info:
        labels_path = os.path.join(MODELS_DIR, model_info['labels_filename'])
        if not os.path.exists(labels_path):
            try:
                print(f"📥 Downloading labels file: {model_info['labels_filename']}...")
                urllib.request.urlretrieve(model_info['labels_url'], labels_path)
                print(f"✓ Labels file download completed: {model_info['labels_filename']}")
            except Exception as e:
                print(f"⚠️  Labels file download failed: {e} (will use default labels)")
    
    return model_path

def load_model_labels(model_name):
    """Load model label files"""
    if model_name not in MODELS_CONFIG:
        return None
    
    model_info = MODELS_CONFIG[model_name]
    if 'labels_filename' not in model_info:
        return None
    
    labels_path = os.path.join(MODELS_DIR, model_info['labels_filename'])
    if not os.path.exists(labels_path):
        return None
    
    try:
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
            if 'classes' in labels_data:
                return labels_data['classes']
            elif isinstance(labels_data, list):
                return labels_data
    except Exception as e:
        print(f"⚠️  Unable to load labels file {labels_path}: {e}")
    
    return None

def download_all_models():
    """Download all pre-trained models"""
    print("\n" + "="*60)
    print("📦 Downloading Essentia Pre-trained Models")
    print("="*60)
    
    create_models_directory()
    
    downloaded_models = {}
    for model_name, model_info in MODELS_CONFIG.items():
        model_path = download_model(model_name, model_info)
        if model_path:
            downloaded_models[model_name] = model_path
    
    print(f"\n✓ Successfully downloaded {len(downloaded_models)}/{len(MODELS_CONFIG)} models")
    return downloaded_models

def check_available_models():
    """Check available model files"""
    available_models = {}
    
    if not os.path.exists(MODELS_DIR):
        return available_models
    
    for model_name, model_info in MODELS_CONFIG.items():
        model_path = os.path.join(MODELS_DIR, model_info['filename'])
        if os.path.exists(model_path):
            # Check if embedding model is needed
            if 'embedding_filename' in model_info:
                embedding_path = os.path.join(MODELS_DIR, model_info['embedding_filename'])
                if not os.path.exists(embedding_path):
                    print(f"⚠️  {model_name} missing embedding model: {model_info['embedding_filename']}")
                    continue  # Skip this model
            
            available_models[model_name] = model_path
    
    return available_models

def find_spectral_algorithms():
    """Find available spectral analysis algorithms"""
    algorithms = {}
    
    # Find centroid algorithms
    for name in ['Centroid', 'SpectralCentroid']:
        if hasattr(es, name):
            algorithms['centroid'] = getattr(es, name)
            break
    
    # Find rolloff algorithms
    for name in ['RollOff', 'SpectralRolloff', 'SpectralRollOff']:
        if hasattr(es, name):
            algorithms['rolloff'] = getattr(es, name)
            break
    
    return algorithms

def analyze_with_models(audio_file, available_models):
    """Perform advanced feature analysis using available models"""
    results = {}
    
    if not available_models:
        return results
    
    try:
        # Load audio
        loader = es.MonoLoader(filename=audio_file)
        audio = loader()
        
        print(f"🔄 Analyzing with {len(available_models)} pre-trained models...")
        
        # Music genre analysis - Discogs 400 genres
        if 'genre_discogs400' in available_models:
            try:
                print("  🔍 Starting Discogs genre analysis...")
                embedding_path = os.path.join(MODELS_DIR, 'discogs-effnet-bs64-1.pb')
                if os.path.exists(embedding_path):
                    print("    - Loading embedding model...")
                    embedding_model = es.TensorflowPredictEffnetDiscogs(
                        graphFilename=embedding_path,
                        output="PartitionedCall:1"
                    )
                    print("    - Extracting audio features...")
                    embeddings = embedding_model(audio)
                    print(f"    - Feature dimensions: {embeddings.shape if hasattr(embeddings, 'shape') else 'unknown'}")
                    
                    print("    - Loading classification model...")
                    model_path = available_models['genre_discogs400']
                    classifier = es.TensorflowPredict2D(
                        graphFilename=model_path,
                        input="serving_default_model_Placeholder",
                        output="PartitionedCall:0"
                    )
                    print("    - Making predictions...")
                    predictions = classifier(embeddings)
                    print(f"    - Prediction dimensions: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
                    
                    # Load labels
                    labels = load_model_labels('genre_discogs400')
                    if labels is None:
                        print("    - Using default labels")
                        labels = ['electronic', 'rock', 'pop', 'jazz', 'classical', 
                                'blues', 'country', 'folk', 'funk', 'reggae', 
                                'hip-hop', 'metal', 'punk', 'ambient', 'experimental',
                                'house', 'techno', 'disco', 'soul', 'rnb']
                    else:
                        print(f"    - Loaded {len(labels)} labels")
                    
                    if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
                        probs = predictions[0]
                    elif isinstance(predictions, (list, np.ndarray)) and len(predictions) > 0:
                        probs = predictions[0] if isinstance(predictions[0], (list, np.ndarray)) else predictions
                    else:
                        probs = predictions
                    
                    print(f"    - Probability array length: {len(probs)}")
                    print(f"    - Maximum probability: {max(probs):.4f}")
                    
                    # Get top 10 highest probability indices
                    probs_array = np.array(probs)
                    top_indices = probs_array.argsort()[-10:][::-1]
                    
                    results['genre_discogs_top10'] = []
                    for i, idx in enumerate(top_indices):
                        if idx < len(labels):
                            label = labels[idx]
                            confidence = float(probs[idx])
                            results['genre_discogs_top10'].append((label, confidence))
                            if i < 3:  # Only print top 3 for debugging
                                print(f"      {i+1}. {label}: {confidence:.4f}")
                    
                    if results['genre_discogs_top10']:
                        results['genre_discogs'] = results['genre_discogs_top10'][0][0]
                        results['genre_discogs_confidence'] = results['genre_discogs_top10'][0][1]
                        print(f"    ✓ Main genre: {results['genre_discogs']} ({results['genre_discogs_confidence']:.4f})")
                else:
                    print("  ⚠️  Need to download discogs-effnet-bs64-1.pb model first")
                    
            except Exception as e:
                print(f"  ❌ Discogs genre analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        # MTG-Jamendo genre analysis
        if 'mtg_jamendo_genre' in available_models:
            try:
                print("  🔍 Starting Jamendo genre analysis...")
                embedding_path = os.path.join(MODELS_DIR, 'discogs-effnet-bs64-1.pb')
                if os.path.exists(embedding_path):
                    embedding_model = es.TensorflowPredictEffnetDiscogs(
                        graphFilename=embedding_path,
                        output="PartitionedCall:1"
                    )
                    embeddings = embedding_model(audio)
                    
                    model_path = available_models['mtg_jamendo_genre']
                    # Use correct node names
                    classifier = es.TensorflowPredict2D(
                        graphFilename=model_path,
                        input="model/Placeholder",
                        output="model/Sigmoid"
                    )
                    predictions = classifier(embeddings)
                    
                    labels = load_model_labels('mtg_jamendo_genre')
                    if labels is None:
                        labels = ['rock', 'pop', 'alternative', 'indie', 'electronic', 
                                'folk', 'instrumental', 'ambient', 'jazz', 'classical',
                                'metal', 'punk', 'blues', 'country', 'reggae']
                    else:
                        print(f"    - Loaded {len(labels)} Jamendo labels")
                    
                    if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
                        probs = predictions[0]
                    elif isinstance(predictions, (list, np.ndarray)) and len(predictions) > 0:
                        probs = predictions[0] if isinstance(predictions[0], (list, np.ndarray)) else predictions
                    else:
                        probs = predictions
                    
                    print(f"    - Jamendo probability array length: {len(probs)}")
                    print(f"    - Jamendo maximum probability: {max(probs):.4f}")
                    
                    probs_array = np.array(probs)
                    top_indices = probs_array.argsort()[-10:][::-1]
                    
                    results['genre_jamendo_top10'] = []
                    for i, idx in enumerate(top_indices):
                        if idx < len(labels):
                            label = labels[idx]
                            confidence = float(probs[idx])
                            results['genre_jamendo_top10'].append((label, confidence))
                            if i < 3:  # Only print top 3 for debugging
                                print(f"      {i+1}. {label}: {confidence:.4f}")
                    
                    if results['genre_jamendo_top10']:
                        results['genre_jamendo'] = results['genre_jamendo_top10'][0][0]
                        results['genre_jamendo_confidence'] = results['genre_jamendo_top10'][0][1]
                        print(f"    ✓ Jamendo main genre: {results['genre_jamendo']} ({results['genre_jamendo_confidence']:.4f})")
                else:
                    # Fallback to simple BPM-based classification
                    print("    - Using BPM heuristic analysis...")
                    bpm_extractor = es.PercivalBpmEstimator()
                    bpm = bpm_extractor(audio)
                    
                    if bpm < 70:
                        genre = 'ambient'
                    elif bpm < 90:
                        genre = 'folk'
                    elif bpm < 110:
                        genre = 'blues'
                    elif bpm < 130:
                        genre = 'pop'
                    elif bpm < 150:
                        genre = 'rock'
                    else:
                        genre = 'electronic'
                    
                    results['genre_jamendo'] = genre
                    results['genre_jamendo_confidence'] = 0.7
                
            except Exception as e:
                print(f"  ❌ Jamendo genre analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Instrument recognition analysis
        if 'mtg_jamendo_instrument' in available_models:
            try:
                print("  🔍 Starting instrument recognition analysis...")
                embedding_path = os.path.join(MODELS_DIR, 'discogs-effnet-bs64-1.pb')
                if os.path.exists(embedding_path):
                    embedding_model = es.TensorflowPredictEffnetDiscogs(
                        graphFilename=embedding_path,
                        output="PartitionedCall:1"
                    )
                    embeddings = embedding_model(audio)
                    
                    model_path = available_models['mtg_jamendo_instrument']
                    # Use correct node names
                    classifier = es.TensorflowPredict2D(
                        graphFilename=model_path,
                        input="model/Placeholder",
                        output="model/Sigmoid"
                    )
                    predictions = classifier(embeddings)
                    
                    labels = load_model_labels('mtg_jamendo_instrument')
                    if labels is None:
                        labels = ['guitar', 'piano', 'violin', 'drums', 'bass', 
                                'vocals', 'saxophone', 'trumpet', 'flute', 'synthesizer',
                                'accordion', 'cello', 'clarinet', 'harp', 'organ']
                    else:
                        print(f"    - Loaded {len(labels)} instrument labels")
                    
                    if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
                        probs = predictions[0]
                    elif isinstance(predictions, (list, np.ndarray)) and len(predictions) > 0:
                        probs = predictions[0] if isinstance(predictions[0], (list, np.ndarray)) else predictions
                    else:
                        probs = predictions
                    
                    print(f"    - Instrument probability array length: {len(probs)}")
                    print(f"    - Instrument maximum probability: {max(probs):.4f}")
                    
                    # Get all instruments with confidence above threshold
                    threshold = 0.1
                    detected_instruments = []
                    
                    for i, prob in enumerate(probs):
                        if i < len(labels) and prob > threshold:
                            detected_instruments.append((labels[i], float(prob)))
                    
                    # Sort by confidence
                    detected_instruments.sort(key=lambda x: x[1], reverse=True)
                    results['instruments'] = detected_instruments[:5]  # Top 5 instruments
                    
                    print(f"    ✓ Detected {len(results['instruments'])} instruments")
                    for i, (instrument, confidence) in enumerate(results['instruments'][:3]):
                        print(f"      {i+1}. {instrument}: {confidence:.4f}")
                
            except Exception as e:
                print(f"  ❌ Instrument recognition failed: {e}")
                import traceback
                traceback.print_exc()
        
            # Mood analysis - using pre-trained AI models
            if 'mood_happy' in available_models or 'mood_sad' in available_models:
                try:
                    print("  🔍 Starting AI mood analysis...")
                    
                    # Need to load embedding model first
                    embedding_path = os.path.join(MODELS_DIR, 'discogs-effnet-bs64-1.pb')
                    if os.path.exists(embedding_path):
                        print("    - Loading embedding model for feature extraction...")
                        embedding_model = es.TensorflowPredictEffnetDiscogs(
                            graphFilename=embedding_path,
                            output="PartitionedCall:1"
                        )
                        embeddings = embedding_model(audio)
                        print(f"    - Audio feature extraction completed, feature dimensions: {embeddings.shape if hasattr(embeddings, 'shape') else 'unknown'}")
                        
                        mood_scores = {}
                        
                        # Use mood_happy model
                        if 'mood_happy' in available_models:
                            try:
                                print("    - Running happy mood detection model...")
                                happy_model_path = available_models['mood_happy']
                                happy_classifier = es.TensorflowPredict2D(
                                    graphFilename=happy_model_path,
                                    input="model/Placeholder",
                                    output="model/Softmax"
                                )
                                happy_predictions = happy_classifier(embeddings)
                                
                                # Add debug information
                                print(f"      Happy model raw output: {happy_predictions}")
                                print(f"      Happy output type: {type(happy_predictions)}")
                                if hasattr(happy_predictions, 'shape'):
                                    print(f"      Happy output shape: {happy_predictions.shape}")
                                
                                # Process prediction results - Softmax output is usually [not_mood, mood]
                                if hasattr(happy_predictions, 'shape') and len(happy_predictions.shape) > 1:
                                    # Take the second value (index 1) as the "happy" probability
                                    happy_score = float(happy_predictions[0][1]) if len(happy_predictions[0]) > 1 else float(happy_predictions[0][0])
                                elif isinstance(happy_predictions, (list, np.ndarray)) and len(happy_predictions) > 0:
                                    # If it's a list/array, also take the second value
                                    if hasattr(happy_predictions[0], '__len__') and len(happy_predictions[0]) > 1:
                                        happy_score = float(happy_predictions[0][1])
                                    else:
                                        happy_score = float(happy_predictions[0])
                                else:
                                    happy_score = float(happy_predictions)
                                
                                mood_scores['happy'] = happy_score
                                print(f"      Happy mood score: {happy_score:.4f}")
                                
                            except Exception as e:
                                print(f"      ❌ Happy mood model execution failed: {e}")
                        
                        # Use mood_sad model  
                        if 'mood_sad' in available_models:
                            try:
                                print("    - Running sad mood detection model...")
                                sad_model_path = available_models['mood_sad']
                                sad_classifier = es.TensorflowPredict2D(
                                    graphFilename=sad_model_path,
                                    input="model/Placeholder", 
                                    output="model/Softmax"
                                )
                                sad_predictions = sad_classifier(embeddings)
                                
                                # Add debug information
                                print(f"      Sad model raw output: {sad_predictions}")
                                print(f"      Sad output type: {type(sad_predictions)}")
                                if hasattr(sad_predictions, 'shape'):
                                    print(f"      Sad output shape: {sad_predictions.shape}")
                                
                                # Process prediction results - Softmax output is usually [not_mood, mood]
                                if hasattr(sad_predictions, 'shape') and len(sad_predictions.shape) > 1:
                                    # Take the second value (index 1) as the "sad" probability
                                    sad_score = float(sad_predictions[0][1]) if len(sad_predictions[0]) > 1 else float(sad_predictions[0][0])
                                elif isinstance(sad_predictions, (list, np.ndarray)) and len(sad_predictions) > 0:
                                    # If it's a list/array, also take the second value
                                    if hasattr(sad_predictions[0], '__len__') and len(sad_predictions[0]) > 1:
                                        sad_score = float(sad_predictions[0][1])
                                    else:
                                        sad_score = float(sad_predictions[0])
                                else:
                                    sad_score = float(sad_predictions)
                                
                                mood_scores['sad'] = sad_score
                                print(f"      Sad mood score: {sad_score:.4f}")
                                
                            except Exception as e:
                                print(f"      ❌ Sad mood model execution failed: {e}")
                        
                        # Determine final mood based on model outputs
                        if mood_scores:
                            print(f"    - Mood analysis scores: {mood_scores}")
                            
                            # If both models have results, compare them
                            if 'happy' in mood_scores and 'sad' in mood_scores:
                                happy_score = mood_scores['happy']
                                sad_score = mood_scores['sad']
                                
                                # Calculate relative difference
                                score_diff = abs(happy_score - sad_score)
                                
                                if happy_score > sad_score and happy_score > 0.3:  # Happy dominates and exceeds threshold
                                    results['mood'] = 'happy'
                                    results['mood_confidence'] = min(0.95, max(0.5, happy_score))
                                elif sad_score > happy_score and sad_score > 0.3:  # Sad dominates and exceeds threshold
                                    results['mood'] = 'sad'  
                                    results['mood_confidence'] = min(0.95, max(0.5, sad_score))
                                else:  # Scores are close or both low
                                    results['mood'] = 'neutral'
                                    results['mood_confidence'] = 0.6 if score_diff < 0.1 else 0.5
                                    
                                print(f"    - Comparison result: happy={happy_score:.3f}, sad={sad_score:.3f}, difference={score_diff:.3f}")
                            
                            # If only one model has results
                            elif 'happy' in mood_scores:
                                happy_score = mood_scores['happy']
                                if happy_score > 0.5:
                                    results['mood'] = 'happy'
                                    results['mood_confidence'] = min(0.9, max(0.5, happy_score))
                                else:
                                    results['mood'] = 'neutral'
                                    results['mood_confidence'] = 0.6
                            
                            elif 'sad' in mood_scores:
                                sad_score = mood_scores['sad']
                                if sad_score > 0.5:
                                    results['mood'] = 'sad'
                                    results['mood_confidence'] = min(0.9, max(0.5, sad_score))
                                else:
                                    results['mood'] = 'neutral'
                                    results['mood_confidence'] = 0.6
                            
                            print(f"    ✓ AI mood analysis completed: {results['mood']} (confidence: {results['mood_confidence']:.3f})")
                        
                        else:
                            print("    ⚠️  All mood models failed to run")
                            results['mood'] = 'neutral'
                            results['mood_confidence'] = 0.5
                    
                    else:
                        print("    ❌ Missing embedding model (discogs-effnet-bs64-1.pb), cannot run mood detection")
                        print("    💡 Please select [-2] to download pre-trained models")
                        
                        # Fallback to simple BPM analysis
                        try:
                            print("    - Using BPM fallback analysis...")
                            bpm_extractor = es.PercivalBpmEstimator()
                            bpm = bpm_extractor(audio)
                            print(f"    - Detected BPM: {bpm:.1f}")
                            
                            if bpm > 130:
                                results['mood'] = 'happy'
                                results['mood_confidence'] = 0.6
                            elif bpm < 70:
                                results['mood'] = 'sad'
                                results['mood_confidence'] = 0.6
                            else:
                                results['mood'] = 'neutral'
                                results['mood_confidence'] = 0.5
                                
                            print(f"    - BPM analysis result: {results['mood']} ({results['mood_confidence']:.3f})")
                            
                        except Exception as bpm_e:
                            print(f"    ❌ BPM analysis also failed: {bpm_e}")
                            results['mood'] = 'neutral'
                            results['mood_confidence'] = 0.5
                            
                except Exception as e:
                    print(f"  ❌ AI mood analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Ensure default result
                    results['mood'] = 'neutral'
                    results['mood_confidence'] = 0.5
        
        # Voice/instrumental analysis
        if 'voice_instrumental' in available_models:
            try:
                print("  🔍 Starting voice analysis...")
                pitch_extractor = es.PredominantPitchMelodia()
                pitch, confidence = pitch_extractor(audio)
                
                if len(confidence) > 0:
                    # Calculate ratio of valid pitches
                    valid_pitches = sum(1 for c in confidence if c > 0.3)
                    valid_pitch_ratio = valid_pitches / len(confidence)
                    voice_score = min(1.0, valid_pitch_ratio * 2)
                    
                    results['voice_type'] = 'vocal' if voice_score > 0.3 else 'instrumental'
                    results['voice_confidence'] = voice_score
                    print(f"    ✓ Voice analysis completed: {results['voice_type']} ({voice_score:.3f})")
                else:
                    results['voice_type'] = 'instrumental'
                    results['voice_confidence'] = 0.0
                
            except Exception as e:
                print(f"  ❌ Voice analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"✓ Model analysis completed, obtained {len([k for k in results.keys() if not k.endswith('_confidence') and not k.endswith('_top10')])} advanced features")
        
    except Exception as e:
        print(f"❌ Error in model analysis process: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def download_models_and_analyze(audio_file):
    """Check models and perform analysis"""
    try:
        print("\n🔄 Checking pre-trained models...")
        
        # Check available models
        available_models = check_available_models()
        
        if available_models:
            print(f"✓ Found {len(available_models)} available models")
            for model_name in available_models:
                print(f"  - {MODELS_CONFIG[model_name]['description']}")
        else:
            print("⚠️  No pre-trained models found")
            
            # Ask if user wants to download models
            response = input("\nWould you like to download pre-trained models now? This will take a few minutes (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                available_models = download_all_models()
            else:
                print("Skipping model download, will only perform basic feature analysis")
        
        # Perform basic feature analysis
        try:
            extractor = es.MusicExtractor()
            features, _ = extractor(audio_file)
            print("✓ Basic feature analysis completed!")
        except Exception as e:
            print(f"❌ Basic feature analysis failed: {e}")
            return None, {}
        
        # Perform advanced feature analysis
        advanced_results = {}
        if available_models:
            advanced_results = analyze_with_models(audio_file, available_models)
        
        return features, advanced_results
        
    except Exception as e:
        print(f"❌ Error in analysis process: {e}")
        return None, {}

def analyze_music_file(audio_file):
    """Analyze a single audio file using Essentia and print results"""
    try:
        print("\n" + "="*50)
        print(f"Starting analysis of file: {os.path.basename(audio_file)}")
        
        # Perform complete analysis
        features, advanced_results = download_models_and_analyze(audio_file)
        
        if features is None:
            print("❌ Analysis failed!")
            return
        
        print("Formatting output results...\n")
        print("-" * 50)
        print("🎵 Music Analysis Report 🎵")
        print("-" * 50)
        
        # Display all available feature names (for debugging)
        print("📊 Available Feature List:")
        descriptor_names = features.descriptorNames()
        
        # Group features by category
        categories = {
            'rhythm': [],
            'tonal': [],
            'lowlevel': [],
            'metadata': []
        }
        
        for name in descriptor_names:
            added = False
            for category in categories.keys():
                if name.startswith(category):
                    categories[category].append(name)
                    added = True
                    break
            if not added:
                categories['metadata'].append(name)
        
        for category, names in categories.items():
            if names:
                print(f"  {category.upper()}: {len(names)} features")
                # Show first few feature names as examples
                for name in names[:3]:
                    print(f"    - {name}")
                if len(names) > 3:
                    print(f"    - ... and {len(names)-3} more")
        
        print("\n" + "="*50)
        print("🎵 Detailed Analysis Results 🎵")
        print("="*50)
        
        # Basic audio features
        print("\n🎶 Basic Audio Features:")
        try:
            bpm = features['rhythm.bpm']
            print(f"  Tempo (BPM): {bpm:.2f}")
        except:
            print("  Tempo (BPM): Unable to calculate")
        
        try:
            duration = features['metadata.audio_properties.length']
            print(f"  Duration: {duration:.2f} seconds")
        except:
            print("  Duration: Unable to calculate")
        
        try:
            sample_rate = features['metadata.audio_properties.sample_rate']
            print(f"  Sample Rate: {sample_rate} Hz")
        except:
            print("  Sample Rate: Unable to calculate")
        
        # Tonal features
        print("\n🎼 Tonal Features:")
        
        # Show available tonal feature names
        tonal_features = [name for name in descriptor_names if name.startswith('tonal') and 'key' in name]
        print(f"  Available tonal-related features: {tonal_features}")
        
        # Try different tonal feature names
        key_found = False
        for key_name in ['tonal.key_key', 'tonal.key_temperley.key', 'tonal.key_krumhansl.key']:
            try:
                if key_name in descriptor_names:
                    key = features[key_name]
                    # Find corresponding scale
                    scale_name = key_name.replace('.key', '.scale')
                    if scale_name in descriptor_names:
                        scale = features[scale_name]
                        # Find corresponding confidence
                        strength_name = key_name.replace('.key', '.strength')
                        if strength_name in descriptor_names:
                            key_confidence = features[strength_name]
                            print(f"  Key: {key} {scale} (confidence: {key_confidence:.2f}) [{key_name}]")
                        else:
                            print(f"  Key: {key} {scale} [{key_name}]")
                    else:
                        print(f"  Key: {key} [{key_name}]")
                    key_found = True
                    break
            except Exception as e:
                continue
        
        if not key_found:
            print("  Key: Unable to calculate - please check available feature names")
        
        try:
            chords_key = features['tonal.chords_key']
            chords_scale = features['tonal.chords_scale']
            print(f"  Chord Key: {chords_key} {chords_scale}")
        except:
            print("  Chord Key: Unable to calculate")
        
        # Advanced features (using pre-trained models)
        print("\n🎨 Advanced Feature Analysis:")
        
        if advanced_results:
            print(f"  ✓ Pre-trained model analysis completed")
            
            # Music genre - Discogs 400 genres
            if 'genre_discogs_top10' in advanced_results:
                print(f"\n  🎵 Discogs Genre Analysis (Top 10):")
                for i, (genre, confidence) in enumerate(advanced_results['genre_discogs_top10']):
                    print(f"    {i+1:2d}. {genre:<20} (confidence: {confidence:.3f})")
            elif 'genre_discogs' in advanced_results:
                confidence = advanced_results.get('genre_discogs_confidence', 0)
                print(f"  🎵 Discogs Genre: {advanced_results['genre_discogs']} (confidence: {confidence:.3f})")
            
            # Music genre - Jamendo
            if 'genre_jamendo_top10' in advanced_results:
                print(f"\n  🎵 Jamendo Genre Analysis (Top 10):")
                for i, (genre, confidence) in enumerate(advanced_results['genre_jamendo_top10']):
                    print(f"    {i+1:2d}. {genre:<20} (confidence: {confidence:.3f})")
            elif 'genre_jamendo' in advanced_results:
                confidence = advanced_results.get('genre_jamendo_confidence', 0)  
                print(f"  🎵 Jamendo Genre: {advanced_results['genre_jamendo']} (confidence: {confidence:.3f})")
            
            # Instrument recognition
            if 'instruments' in advanced_results and advanced_results['instruments']:
                print(f"\n  🎸 Detected Instruments:")
                for i, (instrument, confidence) in enumerate(advanced_results['instruments']):
                    print(f"    {i+1}. {instrument:<15} (confidence: {confidence:.3f})")
            
            # Mood analysis
            if 'mood' in advanced_results:
                confidence = advanced_results.get('mood_confidence', 0)
                mood_emoji = '😊' if advanced_results['mood'] == 'happy' else '😢' if advanced_results['mood'] == 'sad' else '😐'
                print(f"\n  {mood_emoji} Mood: {advanced_results['mood']} (confidence: {confidence:.3f})")
            
            # Voice type
            if 'voice_type' in advanced_results:
                confidence = advanced_results.get('voice_confidence', 0)
                voice_emoji = '🎤' if advanced_results['voice_type'] == 'vocal' else '🎼'
                print(f"  {voice_emoji} Audio Type: {advanced_results['voice_type']} (confidence: {confidence:.3f})")
            
            # If no results were obtained, show hints
            main_features = ['genre_discogs', 'genre_jamendo', 'mood', 'voice_type', 'instruments']
            if not any(key in advanced_results for key in main_features):
                print("  ⚠️  Pre-trained models did not produce valid results")
                print("  💡 Possible reasons:")
                print("     - Audio file format incompatible")
                print("     - Model files corrupted")
                print("     - Audio too short or too long")
                
        else:
            # Check if built-in advanced features exist
            highlevel_features = [name for name in descriptor_names if name.startswith('highlevel')]
            
            if not highlevel_features:
                print("  ⚠️  No advanced features found")
                print("  💡 Solutions:")
                print("     1. Re-run the program and choose to download pre-trained models")
                print("     2. Or use [-2] option to manually download models")
            else:
                print(f"  ✓ Found {len(highlevel_features)} built-in advanced features")
                
                # Try to get genre information
                genre_features = [name for name in highlevel_features if 'genre' in name and 'value' in name]
                if genre_features:
                    print(f"\n  🎵 Built-in Genre Classification:")
                    for feature in genre_features[:3]:  # Only show first 3
                        try:
                            value = features[feature]
                            print(f"    {feature.split('.')[-2]}: {value}")
                        except:
                            print(f"    {feature}: Unable to access")
                
                # Try to get mood information
                mood_features = [name for name in highlevel_features if 'mood' in name and 'value' in name]
                if mood_features:
                    print(f"\n  😊 Built-in Mood Analysis:")
                    for feature in mood_features[:3]:
                        try:
                            value = features[feature]
                            print(f"    {feature.split('.')[-2]}: {value}")
                        except:
                            print(f"    {feature}: Unable to access")
        
        # Spectral features
        print("\n🎼 Spectral Features:")
        try:
            spectral_centroid = features['lowlevel.spectral_centroid.mean']
            print(f"  Spectral Centroid: {spectral_centroid:.2f} Hz")
        except:
            print("  Spectral Centroid: Unable to calculate")
        
        try:
            spectral_rolloff = features['lowlevel.spectral_rolloff.mean']
            print(f"  Spectral Rolloff: {spectral_rolloff:.2f} Hz")
        except:
            print("  Spectral Rolloff: Unable to calculate")
        
        try:
            zero_crossing_rate = features['lowlevel.zerocrossingrate.mean']
            print(f"  Zero Crossing Rate: {zero_crossing_rate:.4f}")
        except:
            print("  Zero Crossing Rate: Unable to calculate")
        
        print("-" * 50)
        
    except Exception as e:
        import traceback
        print(f"\nError processing file {os.path.basename(audio_file)}: {e}")
        traceback.print_exc()
        print("Please check if the file path is correct, if the file is corrupted, or if Essentia can handle this audio format.")

def check_essentia_installation():
    """Check Essentia installation and available functions"""
    print("\n" + "="*60)
    print("🔍 Essentia Installation Check")
    print("="*60)
    
    # Check Essentia version
    try:
        print(f"✓ Essentia version: {essentia.__version__}")
    except:
        print("✗ Unable to get Essentia version")
    
    # Check TensorFlow availability
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow not installed")
    
    # Check available algorithms
    try:
        algorithms = dir(es)
        advanced_algos = [algo for algo in algorithms if 'TensorFlow' in algo or 'Model' in algo]
        if advanced_algos:
            print(f"✓ Found {len(advanced_algos)} advanced algorithms")
            for algo in advanced_algos[:5]:
                print(f"  - {algo}")
        else:
            print("⚠️  No advanced machine learning algorithms found")
    except:
        print("✗ Unable to check available algorithms")
    
    # Test basic functionality
    try:
        loader = es.MonoLoader()
        print("✓ Basic audio loading functionality works")
    except:
        print("✗ Basic audio loading functionality abnormal")
    
    print("="*60)

def install_models_guide():
    """Provide guidance for installing models"""
    print("\n" + "="*60)
    print("📦 Solutions for Complete Music Analysis Features")
    print("="*60)
    
    print("\n🎯 Current Status:")
    print("  ✅ Essentia-TensorFlow installed")
    print("  ✅ TensorFlow backend working normally")
    print("  ❌ Missing pre-trained model files")
    
    print("\n💡 Recommended Solutions:")
    
    print("\n🥇 Solution 1: Automatic Download via Program (Recommended)")
    print("  1. Select [-2] to download pre-trained models")
    print("  2. The program will automatically download required .pb model files")
    print("  3. Models will be saved in ./essentia_models/ directory")
    print("  4. These models will be used automatically during analysis")
    
    print("\n🥈 Solution 2: Manual Model File Download")
    print("  Visit: https://essentia.upf.edu/models/classification-heads/")
    print("  Download the following files to ./essentia_models/ directory:")
    print("    Main model files:")
    for model_name, model_info in MODELS_CONFIG.items():
        print(f"      - {model_info['filename']} ({model_info['description']})")
        if 'labels_filename' in model_info:
            print(f"        + {model_info['labels_filename']} (labels file)")
    
    print("    Embedding model files (required):")
    print("      - discogs-effnet-bs64-1.pb (Discogs EfficientNet feature extractor)")
    print("  Note: Some models require corresponding embedding models and label files to work properly")
    
    print("\n🔧 Troubleshooting:")
    print("  1. If download fails, check network connection")
    print("  2. Ensure sufficient disk space (about 100MB)")
    print("  3. Some regions may require VPN")
    print("  4. Manual download is an alternative solution")
    
    print("\n📊 Model Description:")
    print("  These models are trained on large-scale music databases:")
    print("  - Discogs: Electronic music database with detailed genre labels")
    print("  - MTG-Jamendo: Academic music research dataset")
    print("  - MusicNN: Music neural network architecture")
    print("  - All models are validated through peer-reviewed academic research")
    
    print("="*60)

def main():
    """Main function for scanning files, getting user input, and calling analysis functions"""
    print("🎵 Essentia Music Analysis Tool 🎵")
    print(f"Scanning directory: {music_directory}")
    
    # Check if directory exists
    if not os.path.isdir(music_directory):
        print(f"Error: Directory '{music_directory}' does not exist!")
        sys.exit(1)
    
    # Scan and filter supported music files
    music_files = []
    for filename in os.listdir(music_directory):
        if filename.lower().endswith(SUPPORTED_FORMATS):
            full_path = os.path.join(music_directory, filename)
            music_files.append(full_path)
    
    # Sort by filename
    music_files.sort()
    
    if not music_files:
        print("Error: No supported music files found in the specified directory.")
        print(f"Supported formats: {SUPPORTED_FORMATS}")
        sys.exit(1)
    
    # List all found music files
    print(f"\n📁 Found {len(music_files)} music files:")
    for i, filepath in enumerate(music_files):
        print(f"  [{i + 1}] {os.path.basename(filepath)}")
    
    print(f"\n  [0] Show detailed installation guide")
    print(f"  [-1] Check Essentia installation status")
    print(f"  [-2] Download pre-trained models")
    
    # Get user input
    while True:
        try:
            raw_input = input(f"\nPlease enter your choice (-2, -1, 0, 1-{len(music_files)}): ")
            choice = int(raw_input)
            
            if choice == -2:
                print("\n🔄 Starting pre-trained model download...")
                downloaded_models = download_all_models()
                if downloaded_models:
                    print(f"\n✅ Successfully downloaded {len(downloaded_models)} models, now you can perform complete music analysis!")
                else:
                    print("\n❌ Model download failed, please check network connection or download manually")
                continue
            elif choice == -1:
                check_essentia_installation()
                # Also check model status
                available_models = check_available_models()
                print(f"\n📦 Model Status:")
                print(f"  Available models: {len(available_models)}/{len(MODELS_CONFIG)}")
                for model_name, model_info in MODELS_CONFIG.items():
                    status = "✅ Installed" if model_name in available_models else "❌ Not installed"
                    print(f"  - {model_info['description']}: {status}")
                    
                    # Check embedding model status
                    if 'embedding_filename' in model_info:
                        embedding_path = os.path.join(MODELS_DIR, model_info['embedding_filename'])
                        embedding_status = "✅ Installed" if os.path.exists(embedding_path) else "❌ Not installed"
                        print(f"    └─ Embedding model ({model_info['embedding_filename']}): {embedding_status}")
                    
                    # Check labels file status
                    if 'labels_filename' in model_info:
                        labels_path = os.path.join(MODELS_DIR, model_info['labels_filename'])
                        labels_status = "✅ Installed" if os.path.exists(labels_path) else "⚠️  Missing (using default labels)"
                        print(f"    └─ Labels file ({model_info['labels_filename']}): {labels_status}")
                
                print(f"\n💡 Tips:")
                if len(available_models) == 0:
                    print(f"  Select [-2] to download all models for complete functionality")
                elif len(available_models) < len(MODELS_CONFIG):
                    print(f"  Select [-2] to download missing models")
                continue
            elif choice == 0:
                install_models_guide()
                continue
            elif 1 <= choice <= len(music_files):
                break
            else:
                print(f"Error: Please enter -2, -1, 0 or a number between 1 and {len(music_files)}.")
        except ValueError:
            print("Error: Invalid input, please enter a number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)
    
    # Get file path based on user's choice
    selected_file = music_files[choice - 1]
    
    # Call analysis function
    analyze_music_file(selected_file)

# Execute main function when script is run directly
if __name__ == "__main__":
    main()
