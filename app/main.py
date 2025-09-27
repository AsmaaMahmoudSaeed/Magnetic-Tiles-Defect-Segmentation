import asyncio
from io import BytesIO
from pathlib import Path

import numpy as np
from fastai.vision.all import PILImage, load_learner
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import sys
from pathlib import Path
import sys
import base64

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.eval_utils import resize_and_crop_center

BASE_DIR = Path(__file__).resolve().parent
MODEL_PICKEL_PATH = BASE_DIR.parent.joinpath("models", "model_pickle_fastai.pkl").resolve()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def setup_learner():
    learn=load_learner(MODEL_PICKEL_PATH)
    learn.dls.device='cpu'
    return learn
@app.on_event("startup")
async def startup_event():
    global learn
    tasks=[asyncio.ensure_future(setup_learner())]
    learn=(await asyncio.gather(*tasks))[0]

@app.post("/analyze")
async def analyze(image: bytes = File(...)):
  
    img = Image.open(BytesIO(image)).convert("RGB")

    img_cropped = resize_and_crop_center(img)  
    if not isinstance(img_cropped, Image.Image):
        img_cropped = Image.fromarray(np.array(img_cropped))

    overlay = img_cropped.convert("RGBA")  

    img_for_model = PILImage(img_cropped)
    pred, *_ = learn.predict(img_for_model)
    pred = np.array(pred).astype(np.uint8)   

    if pred.ndim == 3:
        pred = np.squeeze(pred)
    h, w = pred.shape  

    mask = np.zeros((h, w, 4), dtype=np.uint8)
    mask[pred == 1] = [255, 0, 0, 120]  
    mask_img = Image.fromarray(mask, mode="RGBA")  

    if overlay.size != mask_img.size:
        mask_img = mask_img.resize(overlay.size, resample=Image.NEAREST)

    combined = Image.alpha_composite(overlay, mask_img)

    buf = BytesIO()
    combined.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    buf2 = BytesIO()
    img_cropped.save(buf2, format="PNG")
    original_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    return {"original": original_b64, "overlay_result": overlay_b64}