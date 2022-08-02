from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from . import routers

app = FastAPI()


@app.get("/", include_in_schema=False)
async def home():
    return RedirectResponse(url="/docs")


app.include_router(routers.vision.router)
