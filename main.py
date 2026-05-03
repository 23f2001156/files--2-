import os
import base64
import uuid
import hashlib
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
CATALOG_DIR = Path("catalog")
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
CATALOG_DIR.mkdir(exist_ok=True)

MAX_SIZE_BYTES = 4 * 1024 * 1024   # 4 MB
MAX_DIMENSION  = 1024               # px
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_CATALOG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


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


def slugify_stem(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean[:40] or "asset"


def save_to_catalog(image_png_bytes: bytes, source_filename: str | None) -> str:
    image_hash = hashlib.sha256(image_png_bytes).hexdigest()[:12]
    existing = next(CATALOG_DIR.glob(f"*_{image_hash}.png"), None)
    if existing:
        return existing.name

    stem = slugify_stem(Path(source_filename or "asset").stem)
    file_name = f"{stem}_{image_hash}.png"
    (CATALOG_DIR / file_name).write_bytes(image_png_bytes)
    return file_name


def catalog_public_item(path: Path) -> dict:
    stat = path.stat()
    return {
        "id": path.name,
        "name": path.stem,
        "url": f"/catalog-files/{path.name}",
        "size": stat.st_size,
        "updated_at": int(stat.st_mtime),
    }


def list_catalog_items() -> list[dict]:
    files = [
        p for p in CATALOG_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_CATALOG_EXTENSIONS
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [catalog_public_item(p) for p in files]


def resolve_catalog_image(image_id: str) -> Path:
    safe_name = Path(image_id or "").name
    if not safe_name or safe_name != image_id:
        raise HTTPException(status_code=400, detail="Invalid catalog image id.")

    path = CATALOG_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Catalog image not found.")
    if path.suffix.lower() not in ALLOWED_CATALOG_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported catalog image format.")

    return path


def build_edit_prompt(user_prompt: str, has_object_reference: bool) -> str:
    """Build a structured prompt for reliable interior edits using industry-grade constraints."""
    reference_instruction = (
        "CRITICAL: Use the second uploaded image as the exact replacement subject asset. "
        "Match its shape perfectly. Place the new object EXACTLY in the same spatial location and footprint as the original furniture piece it is replacing. "
        "Do NOT shift its position, rotate it inappropriately, or place it elsewhere in the room. "
        "Ground it with realistic contact shadows that follow the room's primary light direction. "
        "Extract and apply its exact materiality and textures."
        if has_object_reference
        else "Do not introduce unrelated new furniture, decor, or humans unless explicitly requested."
    )

    return (
        "You are an expert architectural visualization AI and senior interior designer. "
        "Your task is to execute the user's request with strict adherence to photorealism.\n\n"
        "CORE CONSTRAINTS:\n"
        "- POSITION & PLACEMENT: It is ABSOLUTELY VITAL that the new furniture exactly replaces the old furniture's position. The new item MUST strictly occupy the same bounding box and footprint as the previous item. Do not shift it to a different wall or area.\n"
        "- ANCHOR & PRESERVE: Do NOT alter the room's overarching geometry, perspective vanishing points, or original camera angle. The rest of the room must remain identical.\n"
        "- LIGHTING SYNTHESIS: Ensure new objects cast physically accurate shadows, receive correct key light, and respect ambient occlusion.\n"
        "- MATERIALITY: Render with tactile, high-fidelity materiality (micro-imperfections, realistic reflections).\n"
        f"- {reference_instruction}\n"
        "- SCALE & DEPTH: Ensure spatial depth, Z-index overlapping, and ergonomic scale are perfectly maintained.\n"
        "- CLEANLINESS: Output a single realistic edited image with NO text, NO watermarks, and NO borders.\n\n"
        f"USER DIRECTIVE: {user_prompt.strip()}"
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/catalog")
async def get_catalog():
    return {"items": list_catalog_items()}


@app.post("/catalog/upload")
async def upload_catalog_image(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Catalog image must be JPEG, PNG, WebP, or GIF.")

    image_bytes = await image.read()
    image_png = prepare_image(image_bytes)
    file_name = save_to_catalog(image_png, image.filename)
    return {"item": catalog_public_item(CATALOG_DIR / file_name)}


@app.post("/edit")
async def edit_room(
    room_image: UploadFile = File(...),
    prompt: str = Form(...),
    object_image: UploadFile = File(None),
    catalog_image: str = Form(None),
):
    # ── Validate inputs ──────────────────────────────────────────────────────
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if room_image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Room image must be JPEG, PNG, WebP, or GIF.")

    # ── Read & prepare room image ─────────────────────────────────────────────
    room_bytes = await room_image.read()
    room_png   = prepare_image(room_bytes)

    # Save to disk (optional, useful for debugging)
    room_id   = uuid.uuid4().hex
    room_path = UPLOADS_DIR / f"{room_id}_room.png"
    room_path.write_bytes(room_png)

    # ── Build prompt ──────────────────────────────────────────────────────────

    # ── Optional object image ─────────────────────────────────────────────────
    object_png_bytes = None
    if object_image and object_image.filename:
        if object_image.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Object image must be JPEG, PNG, WebP, or GIF.")

        obj_bytes = await object_image.read()
        object_png_bytes = prepare_image(obj_bytes)
        obj_path = UPLOADS_DIR / f"{room_id}_object.png"
        obj_path.write_bytes(object_png_bytes)
        save_to_catalog(object_png_bytes, object_image.filename)
    elif catalog_image:
        catalog_path = resolve_catalog_image(catalog_image)
        object_png_bytes = prepare_image(catalog_path.read_bytes())

    final_prompt = build_edit_prompt(prompt, bool(object_png_bytes))
    # ── Call OpenAI images.edit ───────────────────────────────────────────────
    try:
        room_file_tuple = (room_path.name, room_png, "image/png")

        if object_png_bytes:
            # Pass both images as a list
            obj_path_name = UPLOADS_DIR / f"{room_id}_object.png"
            response = client.images.edit(
                model="gpt-image-1.5",
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
                model="gpt-image-1.5",
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
    results_b64 = []
    for idx, image_data in enumerate(response.data):
        if hasattr(image_data, "b64_json") and image_data.b64_json:
            result_b64 = image_data.b64_json
            result_bytes = base64.b64decode(result_b64)
        elif hasattr(image_data, "url") and image_data.url:
            import requests as req
            result_bytes = req.get(image_data.url, timeout=30).content
            result_b64 = base64.b64encode(result_bytes).decode()
        else:
            continue

        out_path = OUTPUTS_DIR / f"{room_id}_result_{idx+1}.png"
        out_path.write_bytes(result_bytes)
        results_b64.append(result_b64)

    if not results_b64:
        raise HTTPException(status_code=500, detail="No image data returned from OpenAI.")

    return JSONResponse({
        "image_b64": results_b64[0],
        "images_b64": results_b64,
        "format": "png"
    })


# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/catalog-files", StaticFiles(directory="catalog"), name="catalog-files")
app.mount("/", StaticFiles(directory="static", html=True), name="root")
