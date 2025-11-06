import importlib.util
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import pytesseract
import re

# optional sklearn dependency for better label clustering
try:
    from sklearn.cluster import DBSCAN
    _have_dbscan = True
except Exception:
    _have_dbscan = False

# canonical month names + lookup used by multiple modules
MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
month_name_to_idx = {m.lower(): i for i, m in enumerate(MONTH_NAMES)}
for i, m in enumerate(MONTH_NAMES):
    month_name_to_idx[m[:3].lower()] = i

# transformers imports kept here so other modules can call load_models()
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(device):
    deplot_proc = Pix2StructProcessor.from_pretrained("google/deplot")
    deplot_model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot").to(device)
    return deplot_proc, deplot_model


def detect_ticks_classic(img_bgr, *,
                         axis_search_frac=(0.55, 0.95),
                         axis_min_length_frac=0.35,
                         band_half_h_frac=0.02,
                         tick_min_gap_frac=0.03,
                         ocr_label_h_frac=0.12,
                         edge_blur=3):
    # ...existing implementation...
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if edge_blur and edge_blur > 1:
        gray = cv2.GaussianBlur(gray, (edge_blur|1, edge_blur|1), 0)

    # 1) search lower area for a dominant horizontal axis
    y0 = int(h * axis_search_frac[0])
    y1 = int(h * axis_search_frac[1])
    roi = gray[y0:y1, :]
    edges = cv2.Canny(roi, 50, 150)

    axis_y = None
    min_len = int(w * axis_min_length_frac)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=60,
                            minLineLength=min_len, maxLineGap=int(w*0.02))
    if lines is not None and len(lines):
        ys = []
        for x1_, y1_, x2_, y2_ in lines.reshape(-1,4):
            ys.append(int((y1_ + y2_)/2))
        axis_y = int(np.median(ys)) + y0
    else:
        proj = edges.sum(axis=1)
        if proj.max() > 0:
            rel = int(np.argmax(proj))
            axis_y = rel + y0

    if axis_y is None:
        return []

    # 2) Build narrow band around axis and compute vertical projection to find tick candidates
    band_h = max(2, int(h * band_half_h_frac))
    band_top = max(0, axis_y - band_h)
    band_bot = min(h, axis_y + band_h)
    band = gray[band_top:band_bot, :]

    sob = cv2.Sobel(band, cv2.CV_16S, 1, 0, ksize=3)
    sob = cv2.convertScaleAbs(sob)
    proj_v = sob.sum(axis=0).astype(np.float32)
    thr = max(1.0, proj_v.mean() * 1.2)
    peaks = np.where(proj_v > thr)[0]

    if peaks.size == 0:
        _, bw = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cc = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
        stats = cc[2]
        centers = []
        for s in stats[1:]:
            x, y, ww, hh, area = s
            if hh >= max(3, band.shape[0]*0.4) and ww <= max(3, w*0.02):
                centers.append(int(x + ww/2))
        peaks = np.array(sorted(set(centers)), dtype=int)

    if peaks.size == 0:
        return []

    # 3) cluster contiguous peak indices into single tick centers (gap clustering)
    gap_px = max(1, int(w * tick_min_gap_frac))
    clusters = []
    cur = [peaks[0]]
    for p in peaks[1:]:
        if p - cur[-1] <= gap_px:
            cur.append(p)
        else:
            clusters.append(int(np.median(cur)))
            cur = [p]
    clusters.append(int(np.median(cur)))
    tick_xs = [int(c) for c in clusters]

    # 4) OCR labels below axis and associate by proximity
    label_h = int(h * ocr_label_h_frac)
    ly0 = axis_y + 2
    ly1 = min(h, axis_y + 2 + label_h)
    if ly0 >= h or ly0 >= ly1:
        labels = []
    else:
        crop = img_bgr[ly0:ly1, :]
        pil = Image.fromarray(crop[:,:,::-1])
        ocr = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
        labels = []
        for i, txt in enumerate(ocr['text']):
            t = txt.strip()
            if not t:
                continue
            left = ocr['left'][i]
            top = ocr['top'][i]
            width = ocr['width'][i]
            cx = left + width/2
            cy = top + ocr['height'][i]/2 + ly0
            labels.append({"text": t, "cx": cx, "cy": cy, "bbox": (left, top+ly0, left+width, top+ly0+ocr['height'][i])})

    # Cluster/merge OCR boxes horizontally to make label candidates
    label_clusters = []
    if labels:
        xs = np.array([l['cx'] for l in labels]).reshape(-1,1)
        if _have_dbscan:
            db = DBSCAN(eps=max(6, w*0.02), min_samples=1).fit(xs)
            for cid in np.unique(db.labels_):
                members = [labels[i] for i,lab in enumerate(db.labels_) if lab==cid]
                tx = " ".join([m['text'] for m in members])
                cx = np.mean([m['cx'] for m in members])
                bb0 = min([m['bbox'][0] for m in members])
                bb1 = min([m['bbox'][1] for m in members])
                bb2 = max([m['bbox'][2] for m in members])
                bb3 = max([m['bbox'][3] for m in members])
                label_clusters.append({"text": tx, "cx": cx, "bbox": (bb0, bb1, bb2, bb3)})
        else:
            labels_sorted = sorted(labels, key=lambda x: x['cx'])
            cur = [labels_sorted[0]]
            for lb in labels_sorted[1:]:
                if lb['cx'] - cur[-1]['cx'] <= max(6, w*0.02):
                    cur.append(lb)
                else:
                    tx = " ".join([m['text'] for m in cur])
                    cx = np.mean([m['cx'] for m in cur])
                    label_clusters.append({"text": tx, "cx": cx, "bbox": (min([m['bbox'][0] for m in cur]), min([m['bbox'][1] for m in cur]), max([m['bbox'][2] for m in cur]), max([m['bbox'][3] for m in cur]))})
                    cur = [lb]
            if cur:
                tx = " ".join([m['text'] for m in cur])
                cx = np.mean([m['cx'] for m in cur])
                label_clusters.append({"text": tx, "cx": cx, "bbox": (min([m['bbox'][0] for m in cur]), min([m['bbox'][1] for m in cur]), max([m['bbox'][2] for m in cur]), max([m['bbox'][3] for m in cur]))})

    # 5) Associate tick xs to nearest label cluster (within half inter-tick gap)
    if len(tick_xs) > 1:
        inter_gap = np.median(np.diff(sorted(tick_xs)))
    else:
        inter_gap = w * 0.1
    max_assoc_dist = max(8, inter_gap * 0.6)

    results = []
    for tx in tick_xs:
        assoc = None
        if label_clusters:
            dists = [abs(tx - lc['cx']) for lc in label_clusters]
            best_i = int(np.argmin(dists))
            if dists[best_i] <= max_assoc_dist:
                assoc = label_clusters[best_i]['text']
                bbox = label_clusters[best_i]['bbox']
            else:
                assoc = None
                bbox = (tx-3, axis_y-3, tx+3, axis_y+3)
        else:
            assoc = None
            bbox = (tx-3, axis_y-3, tx+3, axis_y+3)
        results.append({"x": int(tx), "label": assoc, "bbox": tuple(map(int, bbox))})

    results = sorted(results, key=lambda r: r['x'])
    return results


def annotate_ticks(img_bgr, ticks, color=(0,0,255), erase_pad=6, sample_h=12):
    # ...existing implementation (unchanged) ...
    out = img_bgr.copy()
    h, w = out.shape[:2]

    axis_candidates = [int(t.get("y_axis")) for t in ticks if t.get("y_axis") is not None]
    axis_y = int(np.median(axis_candidates)) if axis_candidates else None

    if axis_y is None:
        bboxes = [t.get("bbox") or t.get("tick_bbox") for t in ticks if (t.get("bbox") or t.get("tick_bbox"))]
        if bboxes:
            tops = [int(b[1]) for b in bboxes]
            axis_y = max(0, min(tops) - 4)
        else:
            axis_y = None

    for t in ticks:
        bbox = t.get("bbox") or t.get("tick_bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = map(int, bbox)
        x0e = max(0, x0 - erase_pad)
        x1e = min(w, x1 + erase_pad)
        y0e = max(0, y0 - erase_pad)
        y1e = min(h, y1 + erase_pad)

        if axis_y is not None:
            if y0e <= axis_y:
                y0e = axis_y + 1
            if y0e >= y1e:
                continue

        sy1 = y0e
        sy0 = max(0, sy1 - sample_h)
        if axis_y is not None and sy0 <= axis_y < sy1:
            sy0 = max(0, axis_y - sample_h)
            sy1 = axis_y
            if sy0 >= sy1:
                sy0 = max(0, y1e - sample_h)
                sy1 = y1e

        sample = out[sy0:sy1, x0e:x1e] if sy1 > sy0 and x1e > x0e else None
        if sample is not None and sample.size:
            med = np.median(sample.reshape(-1, 3), axis=0).astype(int)
            bg_color = (int(med[0]), int(med[1]), int(med[2]))
        else:
            bg_color = (255, 255, 255)

        cv2.rectangle(out, (x0e, y0e), (x1e, y1e), bg_color, thickness=-1)

    try:
        bboxes = [t.get("bbox") or t.get("tick_bbox") for t in ticks if (t.get("bbox") or t.get("tick_bbox"))]
        if bboxes:
            lefts = [int(b[0]) for b in bboxes]
            rights = [int(b[2]) for b in bboxes]
            tops = [int(b[1]) for b in bboxes]
            bots = [int(b[3]) for b in bboxes]
            band_x0 = max(0, min(lefts) - erase_pad)
            band_x1 = min(w, max(rights) + erase_pad)
            band_y0 = max(0, min(tops) - erase_pad)
            band_y1 = min(h, max(bots) + erase_pad)

            if axis_y is not None and band_y0 <= axis_y <= band_y1:
                band_y0 = axis_y + 1
                if band_y0 >= band_y1:
                    raise ValueError("band would erase axis; skip band erase")

            sy0b = max(0, band_y0 - sample_h)
            sample_band = out[sy0b:band_y0, band_x0:band_x1] if band_y0 > sy0b and band_x1 > band_x0 else None
            if sample_band is not None and sample_band.size:
                med = np.median(sample_band.reshape(-1, 3), axis=0).astype(int)
                bg_color = (int(med[0]), int(med[1]), int(med[2]))
            else:
                bg_color = (255,255,255)
            cv2.rectangle(out, (band_x0, band_y0), (band_x1, band_y1), bg_color, thickness=-1)
    except Exception:
        pass

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    if axis_y is not None:
        baseline_y = min(h - 6, axis_y + 16)
    else:
        bottoms = []
        for t in ticks:
            bbox = t.get("bbox") or t.get("tick_bbox")
            if bbox:
                bottoms.append(int(bbox[3]))
            elif t.get("y_axis") is not None:
                bottoms.append(int(t.get("y_axis")) + 10)
        if bottoms:
            baseline_y = min(h - 6, max(bottoms) + 14)
        else:
            baseline_y = h - 20
    baseline_y = max(12, int(baseline_y))

    for t in ticks:
        lbl = (t.get("label") or "").strip() or str(t.get("x", ""))
        x = int(t.get("x", 0))

        (text_w, text_h), baseline = cv2.getTextSize(lbl, font, font_scale, thickness)

        x_text = x
        x_text = max(2, min(x_text, w - text_w - 2))

        y_text = int(baseline_y)
        if axis_y is not None:
            min_ok = axis_y + text_h + 6
            if y_text - text_h <= axis_y:
                y_text = max(y_text, min_ok)
            else:
                y_text = max(y_text, min_ok)
        else:
            y_text = y_text

        y_text = min(h - 4, y_text)

        pad = 2
        rect_x0 = x_text - pad
        rect_y0 = y_text - text_h - pad
        rect_x1 = x_text + text_w + pad
        rect_y1 = y_text + baseline + pad
        rect_x0 = max(0, rect_x0)
        rect_y0 = max(0, rect_y0)
        rect_x1 = min(w, rect_x1)
        rect_y1 = min(h, rect_y1)

        sy0 = max(0, rect_y0 - sample_h)
        sy1 = rect_y0
        if axis_y is not None and sy0 <= axis_y < sy1:
            sy0 = max(0, axis_y - sample_h)
            sy1 = axis_y
            if sy0 >= sy1:
                sy0 = max(0, rect_y1 - sample_h)
                sy1 = rect_y1

        sample = out[sy0:sy1, rect_x0:rect_x1] if sy1 > sy0 and rect_x1 > rect_x0 else None
        if sample is not None and sample.size:
            med = np.median(sample.reshape(-1, 3), axis=0).astype(int)
            bg_color = (int(med[0]), int(med[1]), int(med[2]))
        else:
            bg_color = (255,255,255)

        cv2.rectangle(out, (rect_x0, rect_y0), (rect_x1, rect_y1), bg_color, thickness=-1)
        cv2.putText(out, lbl, (int(x_text), int(y_text)), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

    return out


def detect_month_index_from_text(s):
    if not s:
        return None
    ss = s.lower()
    mnum = re.search(r'\b(1[0-2]|[1-9])\b', ss)
    if mnum:
        try:
            val = int(mnum.group(0))
            if 1 <= val <= 12:
                return val - 1
        except Exception:
            pass
    for token, idx in month_name_to_idx.items():
        if token in ss:
            return idx
    return None


def fallback_find_months_in_band(img_bgr, axis_y=None, search_frac=(0.65,0.99)):
    h, w = img_bgr.shape[:2]
    if axis_y:
        ly0 = min(h-1, axis_y - 6)
    else:
        ly0 = int(h * search_frac[0])
    ly1 = min(h, int(h * search_frac[1]))
    band = img_bgr[ly0:ly1, :]
    pil = Image.fromarray(band[:,:,::-1])
    ocr = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
    found = []
    for i, txt in enumerate(ocr['text']):
        t = txt.strip()
        if not t:
            continue
        left = ocr['left'][i]
        width = ocr['width'][i]
        cx = left + width/2
        idx = detect_month_index_from_text(t)
        if idx is not None:
            found.append({"x": int(cx), "label": t, "month_idx": idx, "abs_x": int(cx)})
        else:
            tl = t.lower()
            for token, mi in month_name_to_idx.items():
                if token in tl:
                    found.append({"x": int(cx), "label": t, "month_idx": mi, "abs_x": int(cx)})
                    break
    return found, (ly0, ly1)


def _project_root():
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd().resolve()


def _ensure_markdown_generated(md_path):
    md_path = Path(md_path)
    if md_path.exists():
        return True

    project_root = _project_root()
    demo_py = project_root / "MinerU" / "demo" / "demo.py"
    demo_output_dir = project_root / "MinerU" / "demo" / "output"
    demo_pdfs_dir = project_root / "MinerU" / "demo" / "pdfs"

    if not demo_py.exists():
        print("demo.py not found at", demo_py, "- cannot auto-generate markdown.")
        return False

    pdfs = list(demo_pdfs_dir.glob("*.pdf")) if demo_pdfs_dir.exists() else []
    if not pdfs:
        print("No PDFs found under", demo_pdfs_dir, "— cannot run demo.parse_doc.")
        return False

    try:
        spec = importlib.util.spec_from_file_location("mineru_demo", str(demo_py))
        demo_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo_mod)
        demo_mod.parse_doc([p for p in pdfs], output_dir=str(demo_output_dir), backend="pipeline")
        print("Ran demo.parse_doc; waiting for markdown generation.")
        return md_path.exists()
    except Exception as e:
        print("Failed to run demo.parse_doc:", e)
        return False


def _resolve_md_path_input(md_input):
    inp = str(md_input)
    p = Path(inp).expanduser()
    if p.exists() or ("/" in inp) or ("\\" in inp) or inp.startswith("."):
        return p.resolve()
    stem = Path(inp).stem
    project_root = _project_root()
    demo_output_dir = project_root / "MinerU" / "demo" / "output"
    candidate = demo_output_dir / stem / "auto" / f"{stem}.md"
    return candidate