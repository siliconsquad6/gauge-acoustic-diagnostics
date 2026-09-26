"""Pull the REAL drawing of each part out of the maintenance manuals and heat-mark the part on it.

For every machine/part it searches that machine's manuals for the page with a figure (exploded view,
sectional drawing, parts diagram) that labels the part, renders just the figure, and paints a heat
blob where the part's label sits. Output (shown on the dashboard instead of AI art):
  web/cache/<machine>/manual_<part>.png   the figure with the heat mark
  web/cache/<machine>/manual_<part>.json  {title, page, file, labeled}

Run from the gauge folder (CPU only, ~1 min), after get_manuals.sh:
  python app/manual_figures.py
If a pick is wrong, pin it in manuals/figures_override.json, e.g.
  {"pump/impeller": {"file": "pump/peerless_horizontal_centrifugal_iom.pdf", "page": 23}}
"""
import json, re
from pathlib import Path
import fitz  # pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

G = Path(__file__).resolve().parent.parent
MAN, CACHE = G / "manuals", G / "web" / "cache"
SYN = {
    "fan":   {"impeller": ["wheel", "impeller"], "motor": ["motor"], "bearing": ["bearing"], "shaft": ["shaft"]},
    "pump":  {"impeller": ["impeller"], "seal": ["mechanical seal", "seal"], "bearing": ["bearing"], "shaft": ["shaft"], "motor": ["motor"]},
    "valve": {"seat": ["seat", "orifice", "body"], "diaphragm": ["diaphragm"], "plunger": ["core", "plunger"],
              "spring": ["spring"], "coil": ["solenoid", "coil"]},
}
FIG_WORDS = ["exploded", "sectional", "cross section", "cross-section", "assembly", "parts list", "figure", "fig."]
override = json.loads((MAN / "figures_override.json").read_text()) if (MAN / "figures_override.json").exists() else {}
sources = json.loads((MAN / "sources.json").read_text())


def figure_rect(page):
    """Bounding box of the drawing on a page: raster images and/or dense vector line art."""
    pr, rects = page.rect, []
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            if r.width * r.height > 0.04 * pr.width * pr.height:
                rects.append(r)
    draws = [d["rect"] for d in page.get_drawings() if d["rect"].width < 0.9 * pr.width and d["rect"].height < 0.9 * pr.height]
    if len(draws) > 40:
        u = fitz.Rect(draws[0])
        for r in draws[1:]:
            u |= r
        if u.width * u.height > 0.06 * pr.width * pr.height:
            rects.append(u)
    if not rects:
        return None
    u = fitz.Rect(rects[0])
    for r in rects[1:]:
        u |= r
    return u & pr


def best_page(machine, words):
    best = None
    for src in sources:
        if src["machine"] != machine:
            continue
        pdf = MAN / src["file"]
        if not pdf.exists():
            continue
        doc = fitz.open(pdf)
        for pno, page in enumerate(doc):
            fig = figure_rect(page)
            if fig is None:
                continue
            txt = page.get_text().lower()
            hits_in = sum(1 for w in words for r in page.search_for(w) if fig.intersects(r))
            hits_all = sum(txt.count(w) for w in words)
            score = 3 + 3 * min(hits_in, 4) + min(hits_all, 5) + 2 * any(k in txt for k in FIG_WORDS)
            if hits_all == 0:
                continue
            if best is None or score > best[0]:
                best = (score, src, pno)
    return best


def heat(img, centers, label):
    """Soft red/orange heat blobs at the part's label positions + a small tag."""
    W, H = img.size
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    R = int(min(W, H) * 0.09)
    for (x, y) in centers:
        for k, a in enumerate([70, 95, 125, 155, 185]):
            rr = int(R * (1 - k * 0.17))
            col = (255, 60 + k * 25, 20, a)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=col)
    over = over.filter(ImageFilter.GaussianBlur(R * 0.28))
    out = Image.alpha_composite(img.convert("RGBA"), over)
    d = ImageDraw.Draw(out)
    for (x, y) in centers:
        d.ellipse([x - R * 0.55, y - R * 0.55, x + R * 0.55, y + R * 0.55], outline=(220, 30, 20, 255), width=4)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, W // 40))
    except Exception:
        f = ImageFont.load_default()
    tag = f"SUSPECTED FAULT: {label.upper()}"
    x, y = centers[0]
    tw = d.textlength(tag, font=f)
    tx, ty = min(max(8, x - tw / 2), W - tw - 16), max(8, y - R - 40)
    d.rounded_rectangle([tx - 8, ty - 6, tx + tw + 8, ty + f.size + 8], 6, fill=(255, 255, 255, 235), outline=(210, 40, 30, 255), width=3)
    d.text((tx, ty), tag, fill=(200, 30, 20, 255), font=f)
    return out.convert("RGB")


for machine, parts in SYN.items():
    for part, words in parts.items():
        key = f"{machine}/{part}"
        if key in override:
            o = override[key]
            src = next(s for s in sources if s["file"] == o["file"])
            pick = (0, src, int(o["page"]) - 1)
        else:
            pick = best_page(machine, words)
        if not pick:
            print(f"{key}: no labelled figure found in the manuals"); continue
        _, src, pno = pick
        page = fitz.open(MAN / src["file"])[pno]
        fig = figure_rect(page) or page.rect
        clip = fitz.Rect(fig.x0 - 12, fig.y0 - 12, fig.x1 + 12, fig.y1 + 12) & page.rect
        zoom = 1100 / max(clip.width, 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        centers = []
        for w in words:
            for r in page.search_for(w):
                if clip.intersects(r):
                    centers.append((((r.x0 + r.x1) / 2 - clip.x0) * zoom, ((r.y0 + r.y1) / 2 - clip.y0) * zoom))
            if centers:
                break
        img = heat(img, centers[:2], part) if centers else img
        out = CACHE / machine
        out.mkdir(parents=True, exist_ok=True)
        img.save(out / f"manual_{part}.png")
        (out / f"manual_{part}.json").write_text(json.dumps(
            {"title": src["title"], "page": pno + 1, "file": src["file"], "labeled": bool(centers)}))
        print(f"{key}: {src['title']} p.{pno + 1} ({'part marked' if centers else 'figure only, label not found'})")
print("DONE")
