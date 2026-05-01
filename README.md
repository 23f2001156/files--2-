# AI Room Visualizer

A minimal demo web app that uses **OpenAI's `gpt-image-1`** model to edit room photos based on natural language prompts.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key
Edit `.env` and replace the placeholder:
```
OPENAI_API_KEY=sk-...your-key-here...
```

### 3. Run the server
```bash
uvicorn main:app --reload
```

### 4. Open in browser
```
http://localhost:8000/static/index.html
```

---

## Usage

1. **Upload a room photo** — JPEG, PNG, or WebP.
2. *(Optional)* **Upload a replacement object** — e.g. a sofa photo.
3. **Type a prompt** — e.g.
   - *"Remove the sofa from the room"*
   - *"Replace the wooden chair with a modern armchair"*
   - *"Change the wall colour to warm sage green"*
4. Click **Generate AI Edit** and wait ~15–30 seconds.
5. Compare **before / after** and **download** the result.

---

## Project Structure

```
interior-viz-demo/
├── main.py              ← FastAPI backend
├── static/
│   └── index.html       ← Single-page frontend
├── uploads/             ← Temp uploaded files
├── outputs/             ← Saved result images
├── .env                 ← Your OPENAI_API_KEY goes here
└── requirements.txt
```

---

## Estimated Cost

| Action | Approx. cost |
|---|---|
| Single 1024×1024 edit | **~$0.04 – $0.08** |

Pricing depends on input/output token usage with `gpt-image-1`. See [OpenAI pricing](https://openai.com/pricing) for current rates.

---

## Notes

- Images are automatically **converted to PNG** and **resized to ≤ 1024×1024** before being sent to the API.
- Files over 4 MB are resized down iteratively.
- Uploaded and output images are saved to `uploads/` and `outputs/` for debugging; you can safely delete them.
- No ML libraries (torch, onnx, etc.) are used — inference is 100% via the OpenAI API.
