"""
tests/test_api.py
Tests d'intégration de l'API FastAPI — Brain Tumor Staging.

Prérequis : API démarrée sur http://localhost:8000
    cd api && uvicorn main:app --reload --port 8000

Usage :
    python -m pytest tests/test_api.py -v
    python -m pytest tests/test_api.py -v --api-url http://localhost:8000
    python tests/test_api.py           # mode sans pytest
"""

import io
import os
import sys
import random
import base64
import pytest
import requests
from PIL import Image

# -- Configuration ------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60  # secondes


# -- Utilitaires --------------------------------------------------------------
def create_test_image(width=224, height=224, mode="RGB") -> bytes:
    """Génère une image de test synthétique en mémoire."""
    img = Image.new(mode, (width, height),
                    color=(random.randint(0, 255),
                           random.randint(0, 255),
                           random.randint(0, 255)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def find_real_test_image():
    """Cherche une vraie image dans data/Testing/ si disponible."""
    dirs = [
        os.path.join("data", "Testing", "notumor"),
        os.path.join("data", "Testing", "glioma"),
        os.path.join("data", "Testing", "meningioma"),
        os.path.join("data", "Testing", "pituitary"),
        os.path.join("data", "staged_4classes", "stage_0"),
        os.path.join("data", "staged_4classes", "stage_3"),
    ]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(d, fname)
                with open(path, "rb") as f:
                    return f.read(), fname
    return None, None


# ============================================================================
# Tests
# ============================================================================

class TestHealthCheck:
    """GET / — Santé de l'API"""

    def test_health_returns_200(self):
        r = requests.get(API_URL + "/", timeout=TIMEOUT)
        assert r.status_code == 200, "API non disponible (code {})".format(r.status_code)

    def test_health_json_structure(self):
        r = requests.get(API_URL + "/", timeout=TIMEOUT)
        j = r.json()
        assert "status" in j
        assert j["status"] == "ok"
        assert "model_loaded" in j
        assert "num_classes" in j
        assert "autoencoder_loaded" in j
        assert "api_version" in j

    def test_health_num_classes_valid(self):
        r = requests.get(API_URL + "/", timeout=TIMEOUT)
        n = r.json().get("num_classes", 0)
        assert n in (4, 5), "num_classes doit être 4 ou 5, obtenu : {}".format(n)


class TestModelsEndpoints:
    """GET /models/list et GET /model/info"""

    def test_models_list_returns_200(self):
        r = requests.get(API_URL + "/models/list", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_models_list_structure(self):
        r = requests.get(API_URL + "/models/list", timeout=TIMEOUT)
        j = r.json()
        assert "available_models" in j
        assert "active_model" in j
        assert isinstance(j["available_models"], list)

    def test_model_info_returns_200_if_loaded(self):
        # Vérifier d'abord que le modèle est chargé
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")
        r = requests.get(API_URL + "/model/info", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_model_info_structure(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")
        r = requests.get(API_URL + "/model/info", timeout=TIMEOUT)
        j = r.json()
        assert "model_path" in j
        assert "input_shape" in j
        assert "total_params" in j
        assert "num_classes" in j
        assert "stage_labels" in j


class TestPredictEndpoint:
    """POST /predict — Inférence image unique"""

    def test_predict_synthetic_image(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")

        img_bytes = create_test_image(224, 224)
        r = requests.post(
            API_URL + "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200, "Prédit échoué : {}".format(r.text)

    def test_predict_response_structure(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")

        img_bytes = create_test_image()
        r = requests.post(
            API_URL + "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            timeout=TIMEOUT,
        )
        j = r.json()
        required_fields = [
            "stage_id", "stage_label", "clinical_note",
            "confidence", "probabilities", "gradcam_base64",
            "anomaly_score", "requires_review", "review_reason",
            "model_path", "num_classes",
        ]
        for field in required_fields:
            assert field in j, "Champ manquant : {}".format(field)

    def test_predict_probabilities_sum_to_one(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")

        img_bytes = create_test_image()
        r = requests.post(
            API_URL + "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            timeout=TIMEOUT,
        )
        probs = r.json()["probabilities"]
        total = sum(probs)
        assert abs(total - 1.0) < 0.01, "Probabilités ne somment pas à 1 : {:.4f}".format(total)

    def test_predict_confidence_in_range(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")

        img_bytes = create_test_image()
        r = requests.post(
            API_URL + "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            timeout=TIMEOUT,
        )
        conf = r.json()["confidence"]
        assert 0.0 <= conf <= 1.0, "Confiance hors [0,1] : {}".format(conf)

    def test_predict_stage_id_valid(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")
        num_classes = health.get("num_classes", 4)

        img_bytes = create_test_image()
        r = requests.post(
            API_URL + "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            timeout=TIMEOUT,
        )
        sid = r.json()["stage_id"]
        assert 0 <= sid < num_classes, "stage_id invalide : {}".format(sid)

    def test_predict_gradcam_is_valid_base64(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")

        img_bytes = create_test_image()
        r = requests.post(
            API_URL + "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            timeout=TIMEOUT,
        )
        gcb64 = r.json()["gradcam_base64"]
        if gcb64:
            try:
                decoded = base64.b64decode(gcb64)
                assert len(decoded) > 100, "Grad-CAM PNG trop petit"
            except Exception as e:
                pytest.fail("Grad-CAM base64 invalide : {}".format(e))

    def test_predict_real_image_if_available(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")

        img_bytes, fname = find_real_test_image()
        if img_bytes is None:
            pytest.skip("Aucune image réelle disponible dans data/Testing/")

        r = requests.post(
            API_URL + "/predict",
            files={"file": (fname, img_bytes, "image/jpeg")},
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        j = r.json()
        print("\n[Réel] {} -> {} (conf {:.1%})".format(
            fname, j["stage_label"], j["confidence"]))

    def test_predict_invalid_file_returns_error(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")

        r = requests.post(
            API_URL + "/predict",
            files={"file": ("not_an_image.txt", b"this is not an image", "text/plain")},
            timeout=TIMEOUT,
        )
        assert r.status_code in (400, 422, 500), \
            "Devrait rejeter un fichier non-image : code {}".format(r.status_code)


class TestBatchPredictEndpoint:
    """POST /predict/batch — Inférence batch"""

    def test_batch_predict_two_images(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")

        file_tuples = [
            ("files", ("img1.jpg", create_test_image(), "image/jpeg")),
            ("files", ("img2.jpg", create_test_image(), "image/jpeg")),
        ]
        r = requests.post(
            API_URL + "/predict/batch",
            files=file_tuples,
            timeout=TIMEOUT,
        )
        assert r.status_code == 200
        results = r.json()
        assert len(results) == 2

    def test_batch_predict_response_structure(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")

        file_tuples = [
            ("files", ("img1.jpg", create_test_image(), "image/jpeg")),
        ]
        r = requests.post(
            API_URL + "/predict/batch",
            files=file_tuples,
            timeout=TIMEOUT,
        )
        item = r.json()[0]
        for field in ["filename", "stage_id", "stage_label",
                      "confidence", "requires_review"]:
            assert field in item, "Champ manquant : {}".format(field)

    def test_batch_predict_too_many_files_rejected(self):
        health = requests.get(API_URL + "/", timeout=TIMEOUT).json()
        if not health.get("model_loaded"):
            pytest.skip("Aucun modèle chargé")

        file_tuples = [
            ("files", ("img{}.jpg".format(i), create_test_image(), "image/jpeg"))
            for i in range(25)
        ]
        r = requests.post(
            API_URL + "/predict/batch",
            files=file_tuples,
            timeout=120,
        )
        assert r.status_code == 400, \
            "Devrait rejeter >20 images, code obtenu : {}".format(r.status_code)


# ============================================================================
# Mode standalone (sans pytest)
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  TESTS API — Brain Tumor Staging")
    print("  API URL : {}".format(API_URL))
    print("=" * 65)

    tests = [
        ("Health check GET /", lambda: requests.get(API_URL + "/", timeout=10)),
        ("GET /models/list", lambda: requests.get(API_URL + "/models/list", timeout=10)),
        ("GET /model/info", lambda: requests.get(API_URL + "/model/info", timeout=10)),
        ("POST /predict (image synthétique)", lambda: requests.post(
            API_URL + "/predict",
            files={"file": ("test.jpg", create_test_image(), "image/jpeg")},
            timeout=60,
        )),
        ("POST /predict/batch (2 images)", lambda: requests.post(
            API_URL + "/predict/batch",
            files=[("files", ("a.jpg", create_test_image(), "image/jpeg")),
                   ("files", ("b.jpg", create_test_image(), "image/jpeg"))],
            timeout=60,
        )),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            r = fn()
            status = "OK  ({})".format(r.status_code) if r.ok else "FAIL ({})".format(r.status_code)
            print("  [{:4s}] {}".format("OK" if r.ok else "FAIL", name))
            if r.ok:
                passed += 1
            else:
                print("         Réponse : {}".format(r.text[:200]))
                failed += 1
        except requests.exceptions.ConnectionError:
            print("  [SKIP]  {} — API hors ligne".format(name))
            failed += 1
        except Exception as e:
            print("  [ERR ] {} — {}".format(name, e))
            failed += 1

    print("\n" + "=" * 65)
    print("  Résultat : {}/{} tests réussis".format(passed, passed + failed))
    if failed == 0:
        print("  Tous les tests ont réussi !")
    else:
        print("  {} test(s) échoué(s)".format(failed))
    print("=" * 65)
    sys.exit(0 if failed == 0 else 1)

