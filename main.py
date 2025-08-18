from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from stt import shrutam_transcriber # STT and then translation to en
from translators import translate_to_en,translate_to_indic     
from classifier import MobileNetV2Predictor                # crop disease classifier
from rag import CropDiseaseRAG                         # RAG pipeline      
from tts import ImprovedTTSProcessor,initialize_tts    # back-translation and TTS for hi


MODEL_PATH = "/kaggle/input/mobilenetv2/pytorch/default/1/best_finetuned_model.pth"  # Path to your fine-tuned model

pipe = initialize_tts()
predictor = MobileNetV2Predictor(MODEL_PATH)

rag_system = CropDiseaseRAG()

tts_processor = ImprovedTTSProcessor(pipe)


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)



@app.route("/")
def home():
    return render_template("test.html")


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

    # ---------- shrutamSTT ----------
    hindi_text = shrutam_transcriber(audio_path)   # returns Hindi text

    # ---------- Translate to English ----------
    en_text = translate_to_en(hindi_text,"hindi")

    # ---------- Image Classification ----------
    single_result = predictor.predict_single_image(image_path, return_probabilities=True)
    predicted_class = single_result['predicted_class'] # e.g., "Corn___Northern_Leaf_Blight"
    crop_name = predicted_class.split("___")[0]
    disease_name = predicted_class.split("___")[1]   

    # ---------- Build query for RAG ----------
    query = f"""
        crop: {crop_name}
        disease: {disease_name}
        symptoms: {en_text if en_text else ""}
    """

    # ---------- RAG ----------
    rag_response_en = rag_system.search(query, top_k=3)

    # ---------- Translate response back to Hindi ----------
    rag_response_hi = translate_to_indic(rag_response_en,"hindi")

    # ---------- TTS ----------
    tts_audio_path = os.path.join(OUTPUT_FOLDER, "tts_output.wav")
    tts_processor.process_long_text(
            rag_response_hi, 
            output_file=tts_audio_path,
            max_chars=80,
            add_silence_between_chunks=True
    )  # saves Hindi speech

    return jsonify({
        "rag_response_hi": rag_response_hi,
        "audio_url": f"/uploads/{os.path.basename(tts_audio_path)}"
    })


# ----------------- Serve Uploaded Files -----------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
