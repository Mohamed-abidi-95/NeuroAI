import os
import io
import base64
import glob
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import tensorflow as tf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from losses import FocalLoss
from gradcam import compute_gradcam, overlay_gradcam

LABELS_4 = {
    0: "Stage 0 - No tumor (Negative Control)",
    1: "Stage I - Benign (Meningioma, WHO Grade I)",
    2: "Stage II - Local Extension (Pituitary)",
    3: "Stage III/IV - Malignant (Glioma WHO Grade III/IV)",
}

LABELS_5 = {
    0: "Stage 0 - No tumor (Negative Control)",
    1: "Stage I - Benign (Meningioma, WHO Grade I)",
    2: "Stage II - Local Extension (Pituitary)",
    3: "Stage III - Anaplastic (Glioma WHO Grade III)",
    4: "Stage IV - Glioblastoma GBM (WHO Grade IV)",
}

CLINICAL_4 = {
    0: "No mass detected. Routine check-up recommended.",
    1: "Slow-growing benign tumor. Close monitoring advised.",
    2: "Localized tumor. Oncology consultation recommended.",
    3: "Malignant tumor detected. URGENT ONCOLOGICAL MANAGEMENT required.",
}

CLINICAL_5 = {
    0: "No mass detected. Routine check-up recommended.",
    1: "Slow-growing benign tumor. Close monitoring advised.",
    2: "Localized tumor. Oncology consultation recommended.",
    3: "Regional lymph node involvement. Urgent management required.",
    4: "Multi-focal lesions detected. ONCOLOGICAL EMERGENCY.",
}

CONF_THRESH = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
AE_THRESH   = float(os.getenv("AE_THRESHOLD", "0.05"))

def find_best_model():
                                                                                   
    candidates = [
                                                       ,
                                                   ,
                                                    ,
                                        ,
                                         ,
                                              ,
                                         ,
                                        ,
    ]
    env_path = os.getenv("MODEL_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

MODEL_PATH  = find_best_model()
AE_PATH     = os.getenv("AE_MODEL_PATH", "models/autoencoder_final.keras")

print("[.] Chargement du modele de classification...")
classifier  = None
num_classes = 4
img_size    = (128, 128)
stage_labels   = LABELS_4
clinical_notes = CLINICAL_4

if MODEL_PATH:
    try:
        classifier = tf.keras.models.load_model(
            MODEL_PATH, custom_objects={"FocalLoss": FocalLoss})

        out_shape   = classifier.output_shape             
        in_shape    = classifier.input_shape                    
        num_classes = int(out_shape[-1])
        img_size    = (int(in_shape[1]), int(in_shape[2]))
        if num_classes == 5:
            stage_labels   = LABELS_5
            clinical_notes = CLINICAL_5
        print("[OK] Modele charge : {} | {} classes | {}x{}".format(
            MODEL_PATH, num_classes, img_size[0], img_size[1]))
    except Exception as e:
        print("[!] Modele non trouve : {}".format(e))
        classifier = None
else:
    print("[!] Aucun modele disponible. Lancez l'entrainement d'abord.")

print("[.] Chargement de l'auto-encodeur...")
autoencoder = None
if os.path.exists(AE_PATH):
    try:
        autoencoder = tf.keras.models.load_model(AE_PATH)
        print("[OK] Auto-encodeur charge : {}".format(AE_PATH))
    except Exception as e:
        print("[!] Auto-encodeur non charge : {}".format(e))

app = FastAPI(
    title="API Detection & Classification des Stades de Tumeurs",
    description=(

    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    stage_id:        int
    stage_label:     str
    clinical_note:   str
    confidence:      float
    probabilities:   List[float]
    gradcam_base64:  str
    anomaly_score:   float
    requires_review: bool
    review_reason:   str
    model_path:      str
    num_classes:     int

class BatchPredictionResponse(BaseModel):
    filename:    str
    stage_id:    int
    stage_label: str
    confidence:  float
    requires_review: bool

def preprocess_image(file_bytes: bytes) -> np.ndarray:
           
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32)

    if img_size[0] < 224:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_for_ae(img_batch: np.ndarray) -> np.ndarray:
                                                                       
    from PIL import Image as PILImage

    if img_batch.shape[1] != 128:
        resized = []
        for img in img_batch:
            pil = PILImage.fromarray(img.astype(np.uint8) if img.max() > 1.0
                                     else (img * 255).astype(np.uint8))
            pil = pil.resize((128, 128))
            resized.append(np.array(pil, dtype=np.float32) / 255.0)
        return np.array(resized)

    if img_batch.max() > 1.0:
        return img_batch / 255.0
    return img_batch

def array_to_base64_png(img_array: np.ndarray) -> str:
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buffer  = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def run_inference(file_bytes: bytes):
                                                                    
    img_batch  = preprocess_image(file_bytes)
    y_proba    = classifier.predict(img_batch, verbose=0)[0]
    stage_id   = int(np.argmax(y_proba))
    confidence = float(np.max(y_proba))

    try:
        heatmap, _   = compute_gradcam(classifier, img_batch, class_idx=stage_id)
        superimposed = overlay_gradcam(img_batch[0], heatmap)
        gradcam_b64  = array_to_base64_png(superimposed)
    except Exception:
        gradcam_b64 = ""

    anomaly_score = 0.0
    if autoencoder is not None:
        try:
            ae_input = preprocess_for_ae(img_batch)
            recon = autoencoder.predict(ae_input, verbose=0)
            anomaly_score = float(np.mean((ae_input - recon) ** 2))
        except Exception:
            pass

    reasons = []
    if confidence < CONF_THRESH:
        reasons.append("Low confidence ({:.1%} < {:.0%})".format(confidence, CONF_THRESH))
    if anomaly_score > AE_THRESH:
        reasons.append("High anomaly score ({:.5f} > {})".format(anomaly_score, AE_THRESH))

    return {
                  :        stage_id,
                     :     stage_labels.get(stage_id, "Unknown"),
                       :   clinical_notes.get(stage_id, ""),
                    :      round(confidence, 4),
                       :   [round(float(p), 4) for p in y_proba],
                        :  gradcam_b64,
                       :   round(anomaly_score, 6),
                         : len(reasons) > 0,
                       :   " | ".join(reasons) if reasons else "None",
                    :      MODEL_PATH or "none",
                     :     num_classes,
    }

@app.get("/", summary="Health check")
async def health_check():
    return {
                :             "ok",
                      :       classifier is not None,
                    :         MODEL_PATH,
                     :        num_classes,
                  :           "{}x{}".format(*img_size),
                            : autoencoder is not None,
                     :        "2.0.0",
    }

@app.post("/predict", response_model=PredictionResponse, summary="Prediction du stade tumoral")
async def predict(
    file: UploadFile = File(..., description="Image medicale (IRM, CT, histopathologie)")
):
    if classifier is None:
        raise HTTPException(status_code=503, detail="No model loaded. Train the model first.")

    file_bytes = await file.read()
    try:
        result = run_inference(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Inference error: {}".format(str(e)))

    return PredictionResponse(**result)

@app.post("/predict/batch", response_model=List[BatchPredictionResponse],
          summary="Prediction batch (plusieurs images)")
async def predict_batch(
    files: List[UploadFile] = File(..., description="Liste d'images medicales")
):
    if classifier is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Max 20 images per batch.")

    results = []
    for f in files:
        try:
            fb   = await f.read()
            res  = run_inference(fb)
            results.append(BatchPredictionResponse(
                filename=f.filename,
                stage_id=res["stage_id"],
                stage_label=res["stage_label"],
                confidence=res["confidence"],
                requires_review=res["requires_review"],
            ))
        except Exception as e:
            results.append(BatchPredictionResponse(
                filename=f.filename,
                stage_id=-1,
                stage_label="ERROR: {}".format(str(e)),
                confidence=0.0,
                requires_review=True,
            ))
    return results

@app.get("/models/list", summary="Liste des modeles disponibles dans models/")
async def list_models():
    patterns = ["models/*.keras", "models/*.h5"]
    found = []
    for p in patterns:
        found.extend(glob.glob(p))
    return {
                          : sorted(found),
                      :     MODEL_PATH,
                     :      num_classes,
    }

@app.get("/model/info", summary="Informations sur le modele actif")
async def model_info():
    if classifier is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    return {
                    :    MODEL_PATH,
                     :   str(classifier.input_shape),
                      :  str(classifier.output_shape),
                      :  int(classifier.count_params()),
                     :   num_classes,
                  :      "{}x{}".format(*img_size),
                      :  stage_labels,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
