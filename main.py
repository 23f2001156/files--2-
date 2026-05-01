import os
import base64
import uuid
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
import openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment. Please add it to your .env file.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="AI Interior Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

MAX_SIZE_BYTES = 4 * 1024 * 1024   # 4 MB
MAX_DIMENSION  = 1024               # px


def prepare_image(file_bytes: bytes) -> bytes:
    """Convert image to PNG, resize if needed, keep under 4 MB."""
    img = Image.open(BytesIO(file_bytes)).convert("RGBA")

    # Resize if too large
    if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    # If still too large, reduce quality iteratively
    scale = 1.0
    while len(data) > MAX_SIZE_BYTES and scale > 0.2:
        scale -= 0.1
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        buf = BytesIO()
        resized.save(buf, format="PNG")
        data = buf.getvalue()

    return data


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/edit")
async def edit_room(
    room_image: UploadFile = File(...),
    prompt: str = Form(...),
    object_image: UploadFile = File(None),
):
    # ── Validate inputs ──────────────────────────────────────────────────────
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if room_image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Room image must be JPEG, PNG, WebP, or GIF.")

    # ── Read & prepare room image ─────────────────────────────────────────────
    room_bytes = await room_image.read()
    room_png   = prepare_image(room_bytes)

    # Save to disk (optional, useful for debugging)
    room_id   = uuid.uuid4().hex
    room_path = UPLOADS_DIR / f"{room_id}_room.png"
    room_path.write_bytes(room_png)

    # ── Build prompt ──────────────────────────────────────────────────────────
    final_prompt = prompt.strip()

    # ── Optional object image ─────────────────────────────────────────────────
    object_png_bytes = None
    if object_image and object_image.filename:
        obj_bytes        = await object_image.read()
        object_png_bytes = prepare_image(obj_bytes)
        obj_path         = UPLOADS_DIR / f"{room_id}_object.png"
        obj_path.write_bytes(object_png_bytes)
        final_prompt += ". Use the second uploaded image as a visual reference for the replacement object."

    # ── Call OpenAI images.edit ───────────────────────────────────────────────
    try:
        room_file_tuple = (room_path.name, room_png, "image/png")

        if object_png_bytes:
            # Pass both images as a list
            obj_path_name = UPLOADS_DIR / f"{room_id}_object.png"
            response = client.images.edit(
                model="gpt-image-1",
                image=[
                    (room_path.name,         room_png,         "image/png"),
                    (obj_path_name.name,     object_png_bytes, "image/png"),
                ],
                prompt=final_prompt,
                n=1,
                size="1024x1024",
            )
        else:
            response = client.images.edit(
                model="gpt-image-1",
                image=room_file_tuple,
                prompt=final_prompt,
                n=1,
                size="1024x1024",
            )

    except openai.BadRequestError as e:
        raise HTTPException(status_code=400, detail=f"OpenAI rejected the request: {str(e)}")
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key. Check your .env file.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit reached. Please wait and try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    # ── Decode result ─────────────────────────────────────────────────────────
    image_data = response.data[0]

    if hasattr(image_data, "b64_json") and image_data.b64_json:
        result_b64   = image_data.b64_json
        result_bytes = base64.b64decode(result_b64)
    elif hasattr(image_data, "url") and image_data.url:
        import requests as req
        result_bytes = req.get(image_data.url, timeout=30).content
        result_b64   = base64.b64encode(result_bytes).decode()
    else:
        raise HTTPException(status_code=500, detail="No image data returned from OpenAI.")

    # Save output
    out_path = OUTPUTS_DIR / f"{room_id}_result.png"
    out_path.write_bytes(result_bytes)

    return JSONResponse({"image_b64": result_b64, "format": "png"})


# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/", StaticFiles(directory="static", html=True), name="root")
