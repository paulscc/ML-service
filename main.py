from fastapi import FastAPI, UploadFile, File
from fastai.learner import load_learner
from fastai.vision.all import PILImage
import io
import pathlib

# fix WindowsPath pickle issue
pathlib.WindowsPath = pathlib.PosixPath

app = FastAPI()
learn = None


@app.on_event("startup")
def startup_event():
    global learn
    print("Cargando modelo...")
    learn = load_learner("multipetsmodel.pkl", cpu=True)
    print("Modelo cargado")


@app.get("/")
def root():
    return {
        "message": "API funcionando",
        "modelActive": learn is not None
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelActive": learn is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = PILImage.create(io.BytesIO(contents))

        pred, pred_idx, probs = learn.predict(img)

        return {
            "success": True,
            "prediction": str(pred),
            "confidence": float(probs[pred_idx]),
            "classes": list(learn.dls.vocab),
            "modelActive": True
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "modelActive": False
        }