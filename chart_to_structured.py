from pathlib import Path
from PIL import Image
import re
import cv2
import numpy as np
import torch

# import the CV/utility helpers from load_utils (no circular import)
from load_utils import (
    detect_ticks_classic,
    annotate_ticks,
    detect_month_index_from_text,
    fallback_find_months_in_band,
)

def run_deplot(pil_img, deplot_proc, deplot_model, device, max_new_tokens=1024):
    prompt = "Generate underlying data table of the chart below."
    inputs = deplot_proc(images=pil_img, text=prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        preds = deplot_model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = deplot_proc.decode(preds[0], skip_special_tokens=True)
    decoded = decoded.replace("<0x0A>", "\n")
    decoded = re.sub(r"<.*?>", "", decoded)
    return decoded.strip()


def preprocess_for_deplot(pil_img, img_path=None, save_debug=False):
    """
    Create a deplot-friendly image using the classic tick-detection + annotate flow.
    Falls back to original image on failure.
    """
    # local fallback copies of month mapping are OK here but detect_month_index_from_text
    # from load_utils is used above for inference.
    try:
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        ticks = detect_ticks_classic(img_bgr,
                                     axis_search_frac=(0.70, 0.99),
                                     axis_min_length_frac=0.35,
                                     band_half_h_frac=0.03,
                                     tick_min_gap_frac=0.02,
                                     ocr_label_h_frac=0.12,
                                     edge_blur=1)
        if not ticks:
            return pil_img, False

        if isinstance(ticks, list) and ticks:
            non_none_idxs = [i for i, r in enumerate(ticks) if r.get("label")]
            if non_none_idxs:
                s = non_none_idxs[0]
                e = non_none_idxs[-1]
                ticks = ticks[s:e+1]
                n_ticks = len(ticks)
                first_label = (ticks[0].get("label") or "").strip()
                last_label = (ticks[-1].get("label") or "").strip()

                first_idx = detect_month_index_from_text(first_label)
                last_idx = detect_month_index_from_text(last_label)

                start_month_idx = None
                if first_idx is not None:
                    start_month_idx = first_idx
                elif last_idx is not None:
                    start_month_idx = (last_idx - (n_ticks - 1)) % 12

                if start_month_idx is not None:
                    for offset, r in enumerate(ticks):
                        month_idx = (start_month_idx + offset) % 12
                        r["label"] = str(month_idx + 1)
                else:
                    pass
            else:
                axis_candidates = [int(t.get("y_axis")) for t in ticks if t.get("y_axis") is not None]
                axis_y = int(np.median(axis_candidates)) if axis_candidates else None
                found_months, band_coords = fallback_find_months_in_band(img_bgr, axis_y=axis_y)
                if found_months:
                    tick_xs = [r["x"] for r in ticks]
                    ticks = sorted(ticks, key=lambda r: r["x"])
                    found_months = sorted(found_months, key=lambda f: f["x"])
                    n_ticks = len(ticks)
                    first_found = found_months[0]
                    last_found = found_months[-1]
                    start_month_idx = None
                    if abs(first_found["x"] - ticks[0]["x"]) < max(20, (ticks[-1]["x"]-ticks[0]["x"])//(max(1,len(ticks)-1))//2):
                        start_month_idx = first_found["month_idx"]
                    elif abs(last_found["x"] - ticks[-1]["x"]) < max(20, (ticks[-1]["x"]-ticks[0]["x"])//(max(1,len(ticks)-1))//2):
                        start_month_idx = (last_found["month_idx"] - (n_ticks - 1)) % 12
                    else:
                        nearest = min(found_months, key=lambda f: min([abs(f["x"] - tx) for tx in tick_xs]))
                        nearest_tick_idx = int(np.argmin([abs(nearest["x"] - tx) for tx in tick_xs]))
                        start_month_idx = (nearest["month_idx"] - nearest_tick_idx) % 12

                    if start_month_idx is not None:
                        for offset, r in enumerate(ticks):
                            month_idx = (start_month_idx + offset) % 12
                            r["label"] = str(month_idx + 1)

        annotated = annotate_ticks(img_bgr, ticks)
        pil_annot = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        if save_debug and img_path:
            dbg_path = Path(img_path).with_name(Path(img_path).stem + "_debug_ticks.png")
            cv2.imwrite(str(dbg_path), annotated)
        return pil_annot, True
    except Exception as e:
        print("preprocess_for_deplot error:", e)
        return pil_img, False