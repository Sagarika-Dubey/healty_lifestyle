from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
import os
import google.generativeai as genai
import requests
import uuid
import io
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import re
import random

# Load environment variables
load_dotenv()

# Set up API keys
GENAI_API_KEY = os.getenv("Gemini_API")
DEEPGRAM_API_KEY = os.getenv("Deepgram_API")

# Configure Gemini
genai.configure(api_key=GENAI_API_KEY)

def split_text_for_tts(text, max_chars=300):
    """
    Ultra-conservative text splitting for Deepgram TTS to prevent payload errors.
    """
    # Clean up the text first
    text = text.strip()
    if not text:
        return []
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    chunks = []
    
    # First, split by paragraphs (double newlines)
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If paragraph is short enough, use it as is
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        
        # Split long paragraphs by sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If single sentence is too long, split it further
            if len(sentence) > max_chars:
                # Split by commas, semicolons, or other natural breaks
                sub_parts = re.split(r'[,;:]\s+', sentence)
                for part in sub_parts:
                    part = part.strip()
                    if len(part) > max_chars:
                        # Last resort: split by character count with word boundaries
                        words = part.split()
                        temp_chunk = ""
                        for word in words:
                            if len(temp_chunk + " " + word) <= max_chars:
                                temp_chunk = temp_chunk + " " + word if temp_chunk else word
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = word
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                    else:
                        chunks.append(part)
                continue
            
            # Check if adding this sentence would exceed the limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

    # Final validation and cleanup
    validated_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and len(chunk) <= max_chars:
            validated_chunks.append(chunk)
        elif chunk:
            # Emergency split for oversized chunks
            words = chunk.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk + " " + word) <= max_chars:
                    temp_chunk = temp_chunk + " " + word if temp_chunk else word
                else:
                    if temp_chunk:
                        validated_chunks.append(temp_chunk.strip())
                    temp_chunk = word[:max_chars]  # Truncate very long words
            if temp_chunk:
                validated_chunks.append(temp_chunk.strip())
    
    print(f"📝 Split text into {len(validated_chunks)} chunks (max chars per chunk: {max_chars})")
    for i, chunk in enumerate(validated_chunks):
        print(f"   Chunk {i+1}: {len(chunk)} chars - '{chunk[:50]}...'")
    
    return validated_chunks

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def deepgram_speech_to_text(audio_file):
    """
    Use Deepgram API to convert speech to text.
    """
    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&punctuate=true"
    
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
    }
    
    files = {
        'audio': audio_file
    }
    
    try:
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract transcript from Deepgram response
        if 'results' in result and 'channels' in result['results']:
            alternatives = result['results']['channels'][0]['alternatives']
            if alternatives and len(alternatives) > 0:
                return alternatives[0]['transcript']
        
        return ""
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Deepgram STT failed: {str(e)}")
    except KeyError as e:
        raise Exception(f"Unexpected Deepgram response format: {str(e)}")

def get_wellness_response(user_message, session_type="general"):
    """
    Generate wellness AI response from Gemini with specialized prompts.
    """
    
    base_prompt = """
    You are Dr. Serenity, a compassionate AI wellness coach combining psychology expertise and yoga training. 
    You help people heal mentally and emotionally through:
    
    CORE PRINCIPLES:
    - Listen with deep empathy and validation
    - Provide calming, soothing responses
    - Offer practical mental health strategies
    - Suggest gentle yoga practices and mindfulness
    - Promote healthy lifestyle habits
    - Create a safe, non-judgmental space
    
    RESPONSE STYLE:
    - Warm, gentle, and understanding tone
    - Use calming language and positive affirmations
    - Speak as if you're sitting with them in a peaceful space
    - Acknowledge their feelings before offering guidance
    - Keep responses supportive yet professional
    - Keep responses concise but meaningful (MAKE SURE THAT YOU DONT EXCEED MORE THAN 200-300 words for better audio processing)
    """
    
    if session_type == "mental_health":
        specific_prompt = """
        FOCUS: Mental Health & Emotional Wellbeing
        
        Provide:
        1. Emotional validation and active listening responses
        2. Gentle coping strategies (breathing, mindfulness, grounding)
        3. Cognitive reframing techniques when appropriate
        4. Stress management and relaxation methods
        5. Simple yoga poses for emotional balance
        6. Remind them they're not alone and progress takes time
        
        ALWAYS encourage professional help for serious concerns while providing immediate comfort.
        MAKE SURE THAT YOU DONT EXCEED MORE THAN 200-300 words
        """
    
    elif session_type == "yoga_wellness":
        specific_prompt = """
        FOCUS: Yoga Practice & Physical Wellness
        
        Provide:
        1. Gentle yoga sequences for different needs (stress, energy, sleep)
        2. Breathing exercises (pranayama) for mental clarity
        3. Meditation and mindfulness practices
        4. Body awareness and self-care tips
        5. Modifications for different abilities
        6. Connection between physical practice and mental wellbeing
        MAKE SURE THAT YOU DONT EXCEED MORE THAN 200-300 words
        """
    
    elif session_type == "lifestyle_nutrition":
        specific_prompt = """
        FOCUS: Healthy Lifestyle & Nutrition
        
        Provide:
        1. Balanced nutrition guidance for mental clarity
        2. Healthy daily routines that support wellbeing
        3. Sleep hygiene for better mental health
        4. Mindful eating practices
        5. Simple meal ideas that nourish body and mind
        6. Hydration and movement recommendations
        7. Creating sustainable, gentle lifestyle changes
        MAKE SURE THAT YOU DONT EXCEED MORE THAN 200-300 words
        """
    
    else:
        specific_prompt = """
        FOCUS: General Wellness Support
        
        Listen to what they need most and respond with appropriate guidance from:
        - Emotional support and validation
        - Stress relief techniques
        - Gentle movement suggestions
        - Healthy habit formation
        - Mindfulness practices
        MAKE SURE THAT YOU DONT EXCEED MORE THAN 200-300 words
        """
    
    full_prompt = f"{base_prompt}\n\n{specific_prompt}\n\nRemember: Be their gentle guide toward healing and wellness. Always end with encouragement and remind them of their inner strength. Keep response concise for better audio processing. MAKE THE RESPONSE IN NOT MORE THAN 250 WORDS. DONOT EXCEED 250 WORDS."
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([full_prompt, f"User says: {user_message}"])
        return response.text
    except Exception as e:
        raise Exception(f"Failed to generate wellness response: {str(e)}")

def deepgram_text_to_speech(text):
    """
    Use Deepgram TTS API to convert text to speech with ultra-conservative limits.
    """
    max_chars = 250  # Very conservative limit
    
    # Validate and truncate if necessary
    if not text or not text.strip():
        raise Exception("No text provided for TTS")
    
    text = text.strip()
    if len(text) > max_chars:
        print(f"⚠️ Truncating text from {len(text)} to {max_chars} characters")
        # Try to truncate at sentence boundary
        sentences = re.split(r'(?<=[.!?])\s+', text)
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence) <= max_chars:
                truncated += sentence + " "
            else:
                break
        text = truncated.strip() if truncated.strip() else text[:max_chars].strip()

    # Log what we're sending to TTS
    print(f"🎙️ Sending to TTS ({len(text)} chars): '{text[:100]}...'")

    # Using a soothing voice model with conservative parameters
    url = "https://api.deepgram.com/v1/speak?model=aura-luna-en&encoding=mp3"

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "text": text
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        # Detailed error checking
        if response.status_code == 413:
            raise Exception(f"Text payload too large ({len(text)} chars). Deepgram rejected it.")
        elif response.status_code == 400:
            print(f"❌ Bad request. Response: {response.text}")
            raise Exception(f"Invalid text format for Deepgram TTS: {response.text}")
        elif response.status_code == 401:
            raise Exception("Deepgram API authentication failed")
        elif response.status_code == 429:
            raise Exception("Deepgram API rate limit exceeded")
        
        response.raise_for_status()
        
        # Verify we got audio content
        if len(response.content) == 0:
            raise Exception("Received empty audio response from Deepgram TTS")
        
        print(f"✅ Generated audio: {len(response.content)} bytes")
        return response.content
        
    except requests.exceptions.Timeout:
        raise Exception("Deepgram TTS request timed out")
    except requests.exceptions.RequestException as e:
        print(f"❌ Deepgram TTS failed: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"❌ Response body: {e.response.text}")
        raise Exception(f"Deepgram TTS failed: {str(e)}")

def deepgram_text_to_speech_multi(text):
    """
    Split long text and generate combined TTS audio with extensive error handling.
    """
    if not text or not text.strip():
        raise Exception("No text provided for TTS")
    
    print(f"🔍 Starting TTS for text length: {len(text)} characters")
    print(f"📄 Full text being processed: '{text[:200]}...'")
    
    # Use very conservative chunking
    chunks = split_text_for_tts(text, max_chars=250)
    
    if not chunks:
        raise Exception("Failed to split text into chunks")
    
    combined_audio = b""
    successful_chunks = 0
    failed_chunks = []
    chunk_details = []

    print(f"🔍 Processing {len(chunks)} text chunks for TTS")

    for idx, chunk in enumerate(chunks):
        chunk_info = {
            'index': idx + 1,
            'text': chunk,
            'length': len(chunk),
            'status': 'pending'
        }
        
        try:
            print(f"🎙 Processing chunk {idx + 1}/{len(chunks)}: {len(chunk)} characters")
            print(f"📝 Chunk content: '{chunk[:100]}...'")
            
            # Additional safety check
            if len(chunk) > 300:
                print(f"⚠️ Chunk {idx + 1} too long ({len(chunk)} chars), skipping")
                chunk_info['status'] = 'skipped_too_long'
                chunk_info['error'] = f"Too long ({len(chunk)} chars)"
                failed_chunks.append(f"Chunk {idx + 1}: too long")
                chunk_details.append(chunk_info)
                continue
            
            audio = deepgram_text_to_speech(chunk)
            if audio and len(audio) > 0:
                combined_audio += audio
                successful_chunks += 1
                chunk_info['status'] = 'success'
                chunk_info['audio_size'] = len(audio)
                print(f"✅ Chunk {idx + 1} processed successfully ({len(audio)} bytes)")
            else:
                chunk_info['status'] = 'failed_empty'
                chunk_info['error'] = 'Empty audio response'
                failed_chunks.append(f"Chunk {idx + 1}: empty audio")
                
        except Exception as e:
            error_msg = str(e)
            chunk_info['status'] = 'failed'
            chunk_info['error'] = error_msg
            print(f"❌ Failed to generate audio for chunk {idx + 1}: {error_msg}")
            failed_chunks.append(f"Chunk {idx + 1}: {error_msg}")
            
            # Try emergency mini-chunking for oversized chunks
            if "413" in error_msg or "too large" in error_msg.lower() or "payload" in error_msg.lower():
                print(f"🔄 Attempting emergency mini-chunking for chunk {idx + 1}")
                try:
                    mini_chunks = split_text_for_tts(chunk, max_chars=150)
                    mini_success = 0
                    for mini_idx, mini_chunk in enumerate(mini_chunks):
                        try:
                            if len(mini_chunk) <= 200:  # Extra safety
                                mini_audio = deepgram_text_to_speech(mini_chunk)
                                if mini_audio and len(mini_audio) > 0:
                                    combined_audio += mini_audio
                                    mini_success += 1
                                    print(f"✅ Mini-chunk {mini_idx + 1} successful")
                        except Exception as mini_e:
                            print(f"❌ Mini-chunk {mini_idx + 1} failed: {str(mini_e)}")
                    
                    if mini_success > 0:
                        successful_chunks += 0.5  # Partial success
                        chunk_info['status'] = 'partial_success'
                        chunk_info['mini_chunks_success'] = mini_success
                        chunk_info['mini_chunks_total'] = len(mini_chunks)
                        
                except Exception as split_e:
                    print(f"❌ Emergency chunking failed: {str(split_e)}")
            
            continue
        
        chunk_details.append(chunk_info)

    # Log detailed results
    print(f"\n📊 TTS Processing Summary:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Successful: {successful_chunks}")
    print(f"   Failed: {len(failed_chunks)}")
    print(f"   Combined audio size: {len(combined_audio)} bytes")
    
    for detail in chunk_details:
        status_emoji = {
            'success': '✅',
            'failed': '❌',
            'skipped_too_long': '⏭️',
            'failed_empty': '🔇',
            'partial_success': '🟡'
        }.get(detail['status'], '❓')
        
        print(f"   {status_emoji} Chunk {detail['index']}: {detail['length']} chars - {detail['status']}")
        if 'error' in detail:
            print(f"      Error: {detail['error']}")

    if successful_chunks == 0:
        error_summary = '; '.join(failed_chunks[:3])  # Limit error message length
        raise Exception(f"Failed to generate audio for any text chunks. Sample errors: {error_summary}")
    
    print(f"🎵 Final combined audio: {len(combined_audio)} bytes from {successful_chunks} successful chunks")
    return combined_audio

# Flask App Setup
app = Flask(__name__)
from flask_cors import CORS
CORS(app)

# Configuration
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'webm', 'ogg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Store audio responses temporarily in memory
audio_cache = {}

# Store user wellness sessions (in-memory for demo - use database in production)
wellness_sessions = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "AI Wellness Coach - Dr. Serenity"}), 200

@app.route('/wellness_chat', methods=['POST'])
def wellness_chat():
    """
    Main wellness chat endpoint - accepts text or audio input.
    Always returns text response, with optional audio.
    """
    try:
        user_message = None
        session_type = "general"
        
        # Check for text message in JSON
        if request.json:
            user_message = request.json.get('message', '').strip()
            session_type = request.json.get('type', 'general')
        
        # Check for text message in form data
        elif request.form.get("message"):
            user_message = request.form.get("message").strip()
            session_type = request.form.get("type", "general")
        
        # Check if audio file is uploaded
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            
            if audio_file.filename == '':
                return jsonify({"error": "No audio file selected"}), 400
            
            if not allowed_file(audio_file.filename):
                return jsonify({"error": f"Unsupported audio format. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"}), 400
            
            # Check file size
            audio_file.seek(0, os.SEEK_END)
            file_size = audio_file.tell()
            audio_file.seek(0)
            
            if file_size > MAX_FILE_SIZE:
                return jsonify({"error": "Audio file too large. Maximum size: 10MB"}), 400
            
            print("🎧 Converting speech to text...")
            user_message = deepgram_speech_to_text(audio_file)
            
            if not user_message.strip():
                return jsonify({"error": "Could not understand the audio. Please try speaking clearly and try again."}), 400
        
        if not user_message:
            return jsonify({"error": "Please share what's on your mind (text or voice message)"}), 400

        print(f"💭 Processing {session_type} session: {user_message[:100]}...")
        
        # Generate wellness response
        print("🌱 Generating wellness response...")
        response_text = get_wellness_response(user_message, session_type)

        # Always prepare the text response first
        session_id = str(uuid.uuid4())
        response_data = {
            "session_id": session_id,
            "user_message": user_message,
            "response_text": response_text,
            "session_type": session_type,
            "wellness_tip": get_daily_wellness_tip(),
            "audio_available": False,
            "text_sent_to_tts": response_text  # Always show what text is being processed
        }

        # Try to generate audio, but don't fail if it doesn't work
        try:
            print("🎵 Generating calming audio response...")
            audio_content = deepgram_text_to_speech_multi(response_text)
            
            # Store session data with audio
            audio_cache[session_id] = {
                'audio_content': audio_content,
                'response_text': response_text,
                'user_message': user_message,
                'session_type': session_type,
                'timestamp': datetime.now().isoformat()
            }
            
            response_data.update({
                "audio_available": True,
                "audio_url": f"{request.host_url}get_audio/{session_id}",
                "audio_size": len(audio_content)
            })
            print("✨ Wellness response with audio ready")
            
        except Exception as audio_error:
            print(f"⚠️ Audio generation failed: {audio_error}")
            print("📝 Returning text-only response")
            # Store session data without audio
            audio_cache[session_id] = {
                'response_text': response_text,
                'user_message': user_message,
                'session_type': session_type,
                'timestamp': datetime.now().isoformat(),
                'audio_error': str(audio_error)
            }
            response_data['audio_error'] = str(audio_error)

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": f"I'm here to listen. Please try again: {str(e)}"}), 500

@app.route('/yoga_sequence', methods=['POST'])
def yoga_sequence():
    """
    Generate personalized yoga sequence based on user needs.
    Enhanced to always display the text content clearly.
    """
    try:
        data = request.json or {}
        need = data.get('need', 'general')  # stress, energy, sleep, flexibility, general
        duration = data.get('duration', 15)  # minutes
        level = data.get('level', 'beginner')  # beginner, intermediate, advanced
        
        yoga_prompt = f"""
        Create a {duration}-minute yoga sequence for {level} level to help with {need}.
        Keep the response focused and structured (MAKE THE RESPONSE IN NOT MORE THAN 250 words for better audio processing).
        
        Provide:
        1. **Warm-up** (2-3 poses, 3-5 minutes)
        2. **Main sequence** (4-6 poses with breathing cues, 8-10 minutes)
        3. **Cool-down** (2-3 poses, 2-3 minutes)
        4. **Breathing technique** to accompany the practice
        5. **Gentle affirmation** for mental wellbeing
        
        Format as a clear, step-by-step guide with timing for each section.
        Include modifications for different abilities and gentle encouragement.
        Use simple, clear language that flows well when spoken aloud.
        End with a positive, encouraging message.
        MAKE SURE THAT YOU DONT EXCEED MORE THAN 200-300 words
        """
        
        print(f"🧘‍♀️ Generating yoga sequence for: {need}, {duration}min, {level} level")
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(yoga_prompt)
        sequence_text = response.text
        
        print(f"📝 Generated yoga sequence ({len(sequence_text)} chars):")
        print(f"   Preview: {sequence_text[:150]}...")
        
        session_id = str(uuid.uuid4())
        response_data = {
            "session_id": session_id,
            "yoga_sequence": sequence_text,  # Main text content
            "display_text": sequence_text,   # Explicit display text
            "text_sent_to_tts": sequence_text,  # Show what's being sent to TTS
            "duration": duration,
            "level": level,
            "focus": need,
            "audio_available": False,
            "request_params": {
                "need": need,
                "duration": duration,
                "level": level
            }
        }
        
        # Try to generate audio
        try:
            print("🎵 Generating audio for yoga sequence...")
            audio_content = deepgram_text_to_speech_multi(sequence_text)
            audio_cache[session_id] = {
                'audio_content': audio_content,
                'response_text': sequence_text,
                'type': 'yoga_sequence',
                'need': need,
                'duration': duration,
                'level': level,
                'timestamp': datetime.now().isoformat()
            }
            response_data.update({
                "audio_available": True,
                "audio_url": f"{request.host_url}get_audio/{session_id}",
                "audio_size": len(audio_content)
            })
            print("✅ Yoga sequence with audio ready")
            
        except Exception as audio_error:
            print(f"⚠️ Audio generation failed for yoga sequence: {audio_error}")
            audio_cache[session_id] = {
                'response_text': sequence_text,
                'type': 'yoga_sequence',
                'audio_error': str(audio_error),
                'timestamp': datetime.now().isoformat()
            }
            response_data['audio_error'] = str(audio_error)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Yoga sequence error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/nutrition_plan', methods=['POST'])
def nutrition_plan():
    """
    Generate personalized nutrition and wellness routine.
    """
    try:
        data = request.json or {}
        goal = data.get('goal', 'general_wellness')  # mental_clarity, energy, stress_relief, general_wellness
        dietary_preferences = data.get('dietary_preferences', [])  # vegetarian, vegan, gluten_free, etc.
        
        nutrition_prompt = f"""
        Create a gentle, sustainable nutrition and wellness plan focused on {goal}.
        Consider dietary preferences: {', '.join(dietary_preferences) if dietary_preferences else 'none specified'}
        Keep the response structured and practical (aim for 300-400 words for better audio processing).
        
        Provide:
        1. **Daily routine structure** (morning, afternoon, evening)
        2. **Nourishing meal ideas** that support mental wellbeing
        3. **Hydration recommendations**
        4. **Mindful eating practices**
        5. **Gentle movement suggestions** throughout the day
        6. **Sleep and stress management tips**
        7. **Weekly wellness goals** that are achievable
        
        Focus on creating a holistic approach that nurtures both body and mind.
        Keep suggestions gentle, flexible, and sustainable for long-term wellbeing.
        Use simple, clear language that flows well when spoken aloud.
        End with encouraging words about progress and self-compassion.
        MAKE SURE THAT YOU DONT EXCEED MORE THAN 200-300 words
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(nutrition_prompt)
        plan_text = response.text
        
        return jsonify({
            "nutrition_plan": plan_text,
            "display_text": plan_text,  # Explicit display text
            "text_sent_to_tts": plan_text,  # Show what would be sent to TTS
            "goal": goal,
            "dietary_preferences": dietary_preferences,
            "wellness_reminder": "Remember: Small, consistent changes create lasting transformation. Be patient and kind with yourself."
        })
        
    except Exception as e:
        print(f"❌ Nutrition plan error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_audio/<session_id>', methods=['GET'])
def get_audio(session_id):
    """
    Stream the cached audio response.
    """
    try:
        if session_id not in audio_cache:
            return jsonify({"error": "Audio not found or expired"}), 404
        
        cached_data = audio_cache[session_id]
        
        if 'audio_content' not in cached_data:
            return jsonify({"error": "Audio not available for this session"}), 404
            
        audio_content = cached_data['audio_content']
        
        return Response(
            audio_content,
            mimetype="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename=wellness_response_{session_id}.mp3",
                "Content-Length": str(len(audio_content)),
                "Cache-Control": "no-cache",
                "Accept-Ranges": "bytes"
            }
        )
        
    except Exception as e:
        print(f"❌ Audio streaming error: {e}")
        return jsonify({"error": "Failed to serve audio response"}), 500

@app.route('/guided_meditation', methods=['POST'])
def guided_meditation():
    """
    Generate guided meditation sessions.
    """
    try:
        data = request.json or {}
        meditation_type = data.get('type', 'relaxation')  # relaxation, anxiety, sleep, focus, loving_kindness
        duration = data.get('duration', 10)  # minutes
        
        meditation_prompt = f"""
        Create a {duration}-minute guided {meditation_type} meditation script.
        Keep the response concise but complete (aim for 100 words for better audio processing).
        
        Structure:
        1. Gentle introduction and settling in (1-2 minutes)
        2. Breathing awareness and body relaxation
        3. Main meditation focus specific to {meditation_type}
        4. Gentle return to awareness
        5. Closing affirmations
        
        Use calming, slow-paced language with natural pauses.
        Include gentle guidance for the mind when it wanders.
        End with loving-kindness and encouragement.
        Write it as if you're speaking directly to someone in a soothing voice.
        Use simple, clear language that flows well when spoken aloud.
        MAKE SURE THAT YOU DONT EXCEED MORE THAN 200-300 words
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(meditation_prompt)
        meditation_text = response.text
        
        print(f"📝 Generated meditation script ({len(meditation_text)} chars):")
        print(f"   Preview: {meditation_text[:150]}...")
        
        session_id = str(uuid.uuid4())
        response_data = {
            "session_id": session_id,
            "meditation_script": meditation_text,  # Main meditation content
            "display_text": meditation_text,       # Explicit display text
            "text_sent_to_tts": meditation_text,   # Show what's being sent to TTS
            "type": meditation_type,
            "duration": duration,
            "audio_available": False,
            "preparation_tips": [
                "Find a quiet, comfortable space where you won't be disturbed",
                "Sit or lie down in a comfortable position",
                "Close your eyes or soften your gaze",
                "Take three deep breaths to begin settling in"
            ]
        }
        
        # Try to generate audio for the meditation
        try:
            print("🎵 Generating audio for guided meditation...")
            audio_content = deepgram_text_to_speech_multi(meditation_text)
            audio_cache[session_id] = {
                'audio_content': audio_content,
                'response_text': meditation_text,
                'type': 'guided_meditation',
                'meditation_type': meditation_type,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            response_data.update({
                "audio_available": True,
                "audio_url": f"{request.host_url}get_audio/{session_id}",
                "audio_size": len(audio_content)
            })
            print("✅ Guided meditation with audio ready")
            
        except Exception as audio_error:
            print(f"⚠️ Audio generation failed for meditation: {audio_error}")
            audio_cache[session_id] = {
                'response_text': meditation_text,
                'type': 'guided_meditation',
                'audio_error': str(audio_error),
                'timestamp': datetime.now().isoformat()
            }
            response_data['audio_error'] = str(audio_error)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Guided meditation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/breathing_exercise', methods=['POST'])
def breathing_exercise():
    """
    Generate guided breathing exercises for different needs.
    """
    try:
        data = request.json or {}
        exercise_type = data.get('type', '4-7-8')  # 4-7-8, box_breathing, alternate_nostril, belly_breathing
        duration = data.get('duration', 5)  # minutes
        
        breathing_exercises = {
            '4-7-8': {
                'name': '4-7-8 Calming Breath',
                'description': 'Inhale for 4, hold for 7, exhale for 8. Perfect for anxiety and sleep.',
                'instruction': 'Place tongue tip behind upper teeth. Inhale through nose for 4 counts, hold for 7, exhale through mouth for 8 with a whoosh sound.'
            },
            'box_breathing': {
                'name': 'Box Breathing (Square Breathing)',
                'description': 'Equal counts for inhale, hold, exhale, hold. Great for focus and stress relief.',
                'instruction': 'Inhale for 4 counts, hold for 4, exhale for 4, hold empty for 4. Visualize drawing a square with your breath.'
            },
            'alternate_nostril': {
                'name': 'Alternate Nostril Breathing',
                'description': 'Ancient yogic technique to balance the nervous system and clear the mind.',
                'instruction': 'Use right thumb to close right nostril, inhale left. Close left with ring finger, release right, exhale. Inhale right, switch, exhale left.'
            },
            'belly_breathing': {
                'name': 'Deep Belly Breathing',
                'description': 'Activates the relaxation response by engaging the diaphragm.',
                'instruction': 'One hand on chest, one on belly. Breathe so only the belly hand moves. Inhale slowly through nose, exhale through pursed lips.'
            }
        }
        
        selected_exercise = breathing_exercises.get(exercise_type, breathing_exercises['4-7-8'])
        
        breathing_script = f"""
        **{selected_exercise['name']} - {duration} Minute Session**
        
        {selected_exercise['description']}
        
        **Preparation:**
        Find a comfortable seated position with your spine straight but not rigid. Rest your hands gently on your knees or in your lap. Allow your shoulders to soften and release any tension you're holding.
        
        **Technique:**
        {selected_exercise['instruction']}
        
        **Guided Practice:**
        Let's begin together. We'll start with a few natural breaths to settle in... 
        
        Now, let's begin the {selected_exercise['name']} technique. Follow along at your own comfortable pace. If you feel lightheaded at any point, simply return to your natural breathing.
        
        Ready? Here we go...
        
        *Continue this rhythm for the next {duration-2} minutes, breathing with intention and awareness. If your mind wanders, gently guide your attention back to your breath. Each breath is bringing you deeper into a state of calm and centeredness.*
        
        **Closing:**
        Beautiful work. Take a moment to notice how you feel now compared to when we started. Carry this sense of calm and centeredness with you into your day. You can return to this breathing technique whenever you need to find your center.
        
        Remember: Your breath is always available to you as an anchor of peace and stability.
        """
        
        return jsonify({
            "breathing_script": breathing_script,
            "display_text": breathing_script,
            "text_sent_to_tts": breathing_script,
            "exercise_name": selected_exercise['name'],
            "type": exercise_type,
            "duration": duration,
            "benefits": selected_exercise['description'],
            "quick_reminder": "Breathe with intention. Let each breath guide you to greater calm and clarity."
        })
        
    except Exception as e:
        print(f"❌ Breathing exercise error: {e}")
        return jsonify({"error": str(e)}), 500

def get_daily_wellness_tip():
    """
    Generate random daily wellness tips.
    """
    tips = [
        "Remember: Progress, not perfection. Every small step toward wellness matters.",
        "Take three deep breaths right now. Notice how your body feels with each exhale.",
        "Gentle reminder: You are worthy of love, care, and compassion - especially from yourself.",
        "Movement is medicine. Even a 5-minute walk can shift your energy and mood.",
        "Hydration supports both body and mind. Sip some water and notice the nourishment.",
        "Your feelings are valid and temporary. They are visitors, not permanent residents.",
        "Practice the art of saying 'no' to create space for what truly serves your wellbeing.",
        "Gratitude is a gateway to peace. Name three things that brought you joy today.",
        "Rest is productive. Your body and mind need time to restore and rejuvenate.",
        "You have survived 100% of your difficult days so far. You are stronger than you know.",
        "Mindful eating: Chew slowly, taste fully, and appreciate the nourishment you're receiving.",
        "Connect with nature today - even if it's just looking out a window at the sky.",
        "Your breath is your anchor. When life feels overwhelming, return to breathing mindfully.",
        "Boundaries are self-care in action. It's okay to protect your energy and peace.",
        "Laughter truly is medicine. Seek moments of joy and lightness in your day."
    ]
    return random.choice(tips)

@app.route('/wellness_tips', methods=['GET'])
def get_wellness_tips():
    """
    Get multiple wellness tips and inspiration.
    """
    try:
        count = request.args.get('count', 5, type=int)
        tips = [get_daily_wellness_tip() for _ in range(min(count, 10))]  # Max 10 tips
        
        return jsonify({
            "wellness_tips": tips,
            "daily_affirmation": "You are exactly where you need to be in this moment. Trust your journey.",
            "mindful_moment": "Take a pause right now. Notice your breath, feel your feet on the ground, and appreciate this present moment."
        })
        
    except Exception as e:
        print(f"❌ Wellness tips error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/session_history/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """
    Retrieve session history and details.
    """
    try:
        if session_id not in audio_cache:
            return jsonify({"error": "Session not found"}), 404
        
        session_data = audio_cache[session_id]
        
        # Remove audio content from response to keep it lightweight
        response_data = {
            "session_id": session_id,
            "response_text": session_data.get('response_text', ''),
            "user_message": session_data.get('user_message', ''),
            "session_type": session_data.get('session_type', 'general'),
            "timestamp": session_data.get('timestamp', ''),
            "audio_available": 'audio_content' in session_data,
            "type": session_data.get('type', 'wellness_chat')
        }
        
        # Add specific data based on session type
        if session_data.get('type') == 'yoga_sequence':
            response_data.update({
                "need": session_data.get('need'),
                "duration": session_data.get('duration'),
                "level": session_data.get('level')
            })
        elif session_data.get('type') == 'guided_meditation':
            response_data.update({
                "meditation_type": session_data.get('meditation_type'),
                "duration": session_data.get('duration')
            })
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Session history error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear_cache', methods=['POST'])
def clear_audio_cache():
    """
    Clear the audio cache (for maintenance).
    """
    try:
        cache_count = len(audio_cache)
        audio_cache.clear()
        return jsonify({
            "message": f"Audio cache cleared successfully",
            "sessions_cleared": cache_count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/app_info', methods=['GET'])
def app_info():
    """
    Get application information and available endpoints.
    """
    return jsonify({
        "app_name": "AI Wellness Coach - Dr. Serenity",
        "version": "1.0.0",
        "description": "Compassionate AI wellness coach combining psychology expertise and yoga training",
        "available_endpoints": {
            "/health": "Health check",
            "/wellness_chat": "Main chat endpoint (text or audio input)",
            "/yoga_sequence": "Generate personalized yoga sequences",
            "/guided_meditation": "Create guided meditation scripts",
            "/breathing_exercise": "Generate breathing exercise guides",
            "/nutrition_plan": "Create nutrition and wellness plans",
            "/wellness_tips": "Get daily wellness tips",
            "/get_audio/{session_id}": "Stream generated audio responses",
            "/session_history/{session_id}": "Retrieve session details",
            "/app_info": "This endpoint"
        },
        "supported_audio_formats": list(ALLOWED_AUDIO_EXTENSIONS),
        "max_file_size": f"{MAX_FILE_SIZE // (1024*1024)}MB",
        "features": [
            "Speech-to-Text (Deepgram)",
            "Text-to-Speech (Deepgram)",
            "AI Wellness Coaching (Gemini)",
            "Personalized Yoga Sequences",
            "Guided Meditations",
            "Breathing Exercises",
            "Nutrition Planning"
        ]
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"}), 413

# Main execution
if __name__ == '__main__':
    print("🌱 Starting AI Wellness Coach - Dr. Serenity")
    print("🔑 Checking API keys...")
    
    if not GENAI_API_KEY:
        print("❌ Gemini API key not found in environment variables")
        exit(1)
    
    if not DEEPGRAM_API_KEY:
        print("❌ Deepgram API key not found in environment variables")
        exit(1)
    
    print("✅ API keys configured")
    print("🚀 Server starting...")
    print("\n📋 Available Endpoints:")
    print("   GET  /health - Health check")
    print("   POST /wellness_chat - Main wellness chat (text/audio)")
    print("   POST /yoga_sequence - Generate yoga sequences")
    print("   POST /guided_meditation - Create meditation scripts")
    print("   POST /breathing_exercise - Generate breathing guides")
    print("   POST /nutrition_plan - Create wellness plans")
    print("   GET  /wellness_tips - Get daily tips")
    print("   GET  /get_audio/{session_id} - Stream audio responses")
    print("   GET  /session_history/{session_id} - Session details")
    print("   GET  /app_info - Application information")
    print("\n🎧 Features:")
    print("   • Speech-to-Text conversion")
    print("   • AI-generated wellness responses")
    print("   • Text-to-Speech with calming voice")
    print("   • Personalized yoga and meditation")
    print("   • Nutrition and lifestyle guidance")
    print("\n💚 Ready to help users on their wellness journey!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)