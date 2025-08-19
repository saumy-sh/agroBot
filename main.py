from flask import Flask, request, jsonify, render_template, send_from_directory
import os
# from stt import shrutam_transcriber # STT 
from translators import translate_to_en,translate_to_indic  #translators for en and hi
from classifier import MobileNetV2Predictor                # crop disease classifier
from rag import CropDiseaseRAG                         # RAG pipeline      
from stt_conformer import stt
# from tts import ImprovedTTSProcessor,initialize_tts    
from llm import call_llm                               #qwen llm
from tts import tts                                    #tts
# from rag_test import answer_queries
# Try to import pydub, but handle if FFmpeg is not available
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception as e:
    print(f"Warning: pydub/FFmpeg not fully available: {e}")
    PYDUB_AVAILABLE = False

MODEL_PATH = "models/best_finetuned_model.pth"  # Path to your fine-tuned model

# diseases mapping
mapping = [
    {"label": "Corn___Common_Rust", "crop": "corn", "disease": "common rust"},
    {"label": "Corn___Gray_Leaf_Spot", "crop": "corn", "disease": "gray leaf spot"},
    {"label": "Corn___Healthy", "crop": "corn", "disease": "healthy"},
    {"label": "Corn___Northern_Leaf_Blight", "crop": "corn", "disease": "northern leaf blight"},
    {"label": "Potato___Early_Blight", "crop": "potato", "disease": "early blight"},
    {"label": "Potato___Healthy", "crop": "potato", "disease": "healthy"},
    {"label": "Potato___Late_Blight", "crop": "potato", "disease": "late blight"},
    {"label": "Rice___Brown_Spot", "crop": "rice", "disease": "brown spot"},
    {"label": "Rice___Healthy", "crop": "rice", "disease": "healthy"},
    {"label": "Rice___Leaf_Blast", "crop": "rice", "disease": "leaf blast"},
    {"label": "Rice___Neck_Blast", "crop": "rice", "disease": "neck blast"},
    {"label": "Wheat___Brown_Rust", "crop": "wheat", "disease": "brown rust"},
    {"label": "Wheat___Healthy", "crop": "wheat", "disease": "healthy"},
    {"label": "Wheat___Yellow_Rust", "crop": "wheat", "disease": "yellow rust"},
    {"label": "Sugarcane__Red_Rot", "crop": "sugarcane", "disease": "red rot"},
    {"label": "Sugarcane__Healthy", "crop": "sugarcane", "disease": "healthy"},
    {"label": "Sugarcane__Bacterial Blight", "crop": "sugarcane", "disease": "bacterial blight"}
]



predictor = MobileNetV2Predictor(MODEL_PATH)

rag_system = CropDiseaseRAG()

# tts_processor = ImprovedTTSProcessor(pipe)

print("initialised models")
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def convert_audio_to_wav(audio_path):
    """
    Convert audio file to WAV format with error handling
    Returns the converted file path or original path if conversion fails
    """
    if not PYDUB_AVAILABLE:
        print("Warning: Cannot convert audio format - FFmpeg not available. Using original file.")
        return audio_path
    
    if audio_path.endswith(".webm") or audio_path.endswith(".ogg"):
        wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
        try:
            # Attempt to convert using pydub
            audio = AudioSegment.from_file(audio_path)
            audio.export(wav_path, format="wav")
            print(f"Successfully converted {audio_path} to {wav_path}")
            return wav_path
        except FileNotFoundError as e:
            if "ffprobe" in str(e) or "ffmpeg" in str(e):
                print(f"FFmpeg not found. Please install FFmpeg to handle WebM/OGG audio files.")
                print("For now, trying to use original file...")
                return audio_path
            else:
                raise e
        except Exception as e:
            print(f"Audio conversion failed: {e}")
            print("Using original audio file...")
            return audio_path
    
    return audio_path


@app.route("/")
def home():
    return render_template("test3.html")


# ----------------- MAIN PIPELINE -----------------
@app.route("/process", methods=["POST"])
def process():
    """
    Complete pipeline:
    1. Receive audio + image
    2. STT → Hindi text
    3. Translate Hindi → English
    4. Classify image
    5. Send query + classification result to RAG
    6. Translate RAG response EN → HI
    7. TTS (Hindi) → return audio + text
    """

    if "audio" not in request.files or "image" not in request.files:
        return jsonify({"error": "Both audio and image required"}), 400

    # Save files
    audio_file = request.files["audio"]
    image_file = request.files["image"]

    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)

    audio_file.save(audio_path)
    image_file.save(image_path)
    
    # Convert audio to WAV if needed (with error handling)
    audio_path = convert_audio_to_wav(audio_path)

    try:
        # ---------- shrutamSTT ----------
        # hindi_text = shrutam_transcriber(audio_path) 
        hindi_text = stt(audio_path)
        print("##############")
        # print(hindi_text)  
        print("1")
        
        # ---------- Translate to English ----------
        en_text = translate_to_en(hindi_text,"hindi")
        print(en_text)
        print("######################")
        print("2")
        
        # ---------- Image Classification ----------
        single_result = predictor.predict_single_image(image_path, return_probabilities=False)
        
        predicted_class = single_result['predicted_class'] # e.g., "Corn___Northern_Leaf_Blight"
        
        print(predicted_class)
        print("#######################################")
        print("####################################")
        print("##################################")
        
        # ---------- Build query for RAG ----------
        query = f"""
            crop_disease: {predicted_class}
        """

        # # ---------- RAG ----------
        rag_response_en = rag_system.search(query, top_k=3)
        print(rag_response_en)
        clean_response = [
            {
                "crop": item["crop"],
                "disease": item["disease"],
                "symptoms": item["symptoms"],
                "treatment": item["treatment"]
            }
            for item in rag_response_en
        ]
        print("4")

        llm_response = call_llm(clean_response[0],en_text)
        # llm_response = answer_queries(predicted_class)
        print("#######################################################")
        print(llm_response)
        print("#######################################################")
        # ---------- Translate response back to Hindi ----------
        rag_response_hi = translate_to_indic(llm_response,"hindi")
        print("5")

        # ---------- TTS ----------
        tts_audio_path = os.path.join(OUTPUT_FOLDER, "tts_output.wav")
        # tts_processor.process_long_text(
        #         rag_response_hi, 
        #         output_file=tts_audio_path,
        #         max_chars=80,
        #         add_silence_between_chunks=True
        # )
        tts(rag_response_hi)
        print("6")  # saves Hindi speech
        print(rag_response_hi)
        
        return jsonify({
            "rag_response_hi": rag_response_hi,
            "audio_url": f"/output/{os.path.basename(tts_audio_path)}"
        })
        
    except Exception as e:
        print(f"Processing error: {e}")
        return jsonify({
            "error": f"Processing failed: {str(e)}",
            "rag_response_hi": "प्रोसेसिंग में त्रुटि हुई।",
            "audio_url": None
        }), 500


# ----------------- Serve Uploaded Files -----------------
# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)