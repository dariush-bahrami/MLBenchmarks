import time

import numpy as np
import torch
from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image
from pydantic import BaseModel

from ..dependencies import MLModelsDict, ml_models_provider

router = APIRouter(tags=["vision"])


class Time(BaseModel):
    mean: float
    std: float


class ClassifierResponseModel(BaseModel):
    label: str
    model: str
    device: str
    time: Time


@router.post("/classifier/", response_model=ClassifierResponseModel)
async def classifier(
    image_file: UploadFile = File(...),
    runs: int = 3,
    models: MLModelsDict = Depends(ml_models_provider),
):
    image = Image.open(image_file.file)
    device = models["vision"]["classifier"].device
    model_name = models["vision"]["classifier"].model_name
    times = []
    for _ in range(runs):
        tick = time.perf_counter()
        label = models["vision"]["classifier"](image)
        tock = time.perf_counter()
        times.append(tock - tick)
    times = np.array(times)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = device.type
    response = {
        "label": label,
        "model": model_name,
        "device": device_name,
        "time": Time(mean=times.mean(), std=times.std()),
    }
    return response
