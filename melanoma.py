"""
Melanoma lesion analysis with OpenCV (ABCD-rule feature extraction).

EDUCATIONAL / RESEARCH TOOL ONLY - NOT A MEDICAL DIAGNOSIS.

Engineering pipeline:
  1. Load + resize            -> consistent scale
  2. Grayscale conversion     -> intensity channel for segmentation
  3. Hair removal (DullRazor) -> blackhat morphology + inpainting
  4. Denoise                  -> Gaussian blur
  5. Contrast enhancement     -> CLAHE
  6. Segmentation             -> Otsu threshold (lesion darker than skin)
  7. Morphological cleanup    -> open/close + keep largest central blob
  8. Contour extraction       -> boundary, area, perimeter
  9. Features (ABCD):
       A  Asymmetry  - mirror overlap along principal axes
       B  Border     - compactness / irregularity index
       C  Color      - number of suspicious colors present
       D  Diameter   - max Feret diameter (pixels -> mm if scale known)
 10. TDS score (Stolz)        -> 1.3*A + 0.1*B + 0.5*C + 0.5*D
 11. Visualization of every step

Usage:
  python melanoma.py image.jpg [--px-per-mm 10] [--out results] [--no-show]
"""
import argparse
import os

import cv2
import numpy as np


# ---------------------------------------------------------------- preprocessing
def resize(img, max_side=600):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, hair_mask, 3, cv2.INPAINT_TELEA), hair_mask


def preprocess(img):
    clean, hair_mask = remove_hair(img)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)
    return clean, gray, hair_mask, clahe


# ---------------------------------------------------------------- validation
class NoLesionError(Exception):
    """Raised when the image does not look like a mole on skin."""

    def __init__(self, reasons):
        self.reasons = reasons
        super().__init__("Mole not detected: " + "; ".join(reasons))


def edge_zone(fov):
    """Pixels on the image border or touching the dermoscope's black frame."""
    zone = cv2.dilate(cv2.bitwise_not(fov), np.ones((7, 7), np.uint8))
    zone[:3, :] = zone[-3:, :] = 255
    zone[:, :3] = zone[:, -3:] = 255
    return zone


def edge_fraction(contour, zone):
    pts = contour.reshape(-1, 2)
    return float((zone[pts[:, 1], pts[:, 0]] > 0).mean())


def validate_lesion(img, gray, mask, contour, fov):
    """Sanity checks that the segmented blob is a plausible mole on skin."""
    reasons = []

    area_ratio = cv2.contourArea(contour) / max(np.count_nonzero(fov), 1)
    if area_ratio < 0.005:
        reasons.append(f"dark region too small ({area_ratio:.1%} of image)")
    elif area_ratio > 0.70:
        reasons.append(f"dark region fills the image ({area_ratio:.0%}), no skin around it")

    # surrounding ring of 'skin' around the lesion
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ring = cv2.subtract(cv2.dilate(mask, k, iterations=2), cv2.dilate(mask, k))
    ring = cv2.bitwise_and(ring, fov)
    if np.count_nonzero(ring) < 100:
        reasons.append("no surrounding skin visible")
    else:
        contrast = float(gray[ring > 0].mean() - gray[mask > 0].mean())
        if contrast < 20:
            reasons.append(f"lesion not darker than its surroundings (contrast {contrast:.0f})")
        # Dermoscope lighting makes skin look pink, beige, gray or even bluish, so a strict
        # skin-tone range fails. Instead reject surroundings that are STRONGLY green/blue/purple
        # (grass, sky, objects), which skin never is.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[ring > 0]
        vivid = ((hsv[:, 1] > 110) & (hsv[:, 0] >= 35) & (hsv[:, 0] <= 150)).mean()
        if vivid > 0.40:
            reasons.append(f"surrounding area does not look like skin ({vivid:.0%} strongly green/blue)")

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = cv2.contourArea(contour) / hull_area if hull_area else 0
    if solidity < 0.60:
        reasons.append(f"shape too fragmented for a mole (solidity {solidity:.2f})")

    if edge_fraction(contour, edge_zone(fov)) > 0.25:
        reasons.append("region is cut off by the image border")

    if reasons:
        raise NoLesionError(reasons)


# ---------------------------------------------------------------- segmentation
def field_of_view(gray):
    """Mask out the black circular frame that dermoscopes leave around the image."""
    h, w = gray.shape
    dark = (gray < 40).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    fov = np.full_like(gray, 255)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x <= 3 or y <= 3 or x + bw >= w - 3 or y + bh >= h - 3  # some scans have a thin frame
        if touches and area > 0.02 * h * w:
            fov[labels == i] = 0
    # drop the soft shadow at the lens edge
    return cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def segment(enh, fov):
    t, _ = cv2.threshold(enh[fov > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = np.where((enh <= t) & (fov > 0), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=3)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise NoLesionError(["no dark region found"])
    h, w = enh.shape
    center = np.array([w / 2, h / 2])
    zone = edge_zone(fov)

    def score(c):  # prefer large blobs near the centre, penalise blobs on the border/frame
        touches = edge_fraction(c, zone) > 0.10
        m = cv2.moments(c)
        if m["m00"] == 0:
            return -1
        cxy = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
        dist = np.linalg.norm(cxy - center) / np.linalg.norm(center)
        return cv2.contourArea(c) * (1 - dist) * (0.3 if touches else 1.0)

    lesion = max(contours, key=score)
    mask = np.zeros_like(th)
    cv2.drawContours(mask, [lesion], -1, 255, cv2.FILLED)
    return mask, lesion, th


# ---------------------------------------------------------------- ABCD features
def asymmetry(mask, contour):
    """Rotate lesion to its principal axis, flip on both axes, measure non-overlap."""
    (cx, cy), _, angle = cv2.fitEllipse(contour) if len(contour) >= 5 else ((0, 0), 0, 0)
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += w / 2 - cx
    M[1, 2] += h / 2 - cy  # also centre it
    rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    area = max(np.count_nonzero(rot), 1)
    scores = []
    for flip in (0, 1):  # 0: vertical axis flip, 1: horizontal
        f = cv2.flip(rot, flip)
        diff = np.count_nonzero(cv2.bitwise_xor(rot, f))
        scores.append(diff / (2 * area))
    # Stolz A: 0,1,2 asymmetric axes (threshold ~ 0.1 non-overlap)
    a_score = sum(s > 0.10 for s in scores)
    return a_score, scores, rot


def border(contour):
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    compact = perim ** 2 / (4 * np.pi * area) if area else 0  # 1.0 = perfect circle
    hull = cv2.convexHull(contour)
    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) else 0
    # Map irregularity to Stolz B (0-8 segments)
    b_score = int(np.clip(round((compact - 1.0) * 8 + (1 - solidity) * 16), 0, 8))
    return b_score, dict(area=area, perimeter=perim, compactness=compact, solidity=solidity)


COLORS_RGB = {  # reference dermoscopic colors
    "white": (230, 230, 230), "red": (200, 40, 40), "light-brown": (180, 110, 60),
    "dark-brown": (90, 45, 20), "blue-gray": (90, 110, 140), "black": (25, 25, 25),
}


def color(img, mask):
    pix = img[mask > 0][:, ::-1].astype(np.float32)  # BGR->RGB
    if len(pix) == 0:
        return 0, {}
    refs = np.array(list(COLORS_RGB.values()), np.float32)
    d = np.linalg.norm(pix[:, None, :] - refs[None, :, :], axis=2)
    nearest = d.argmin(1)
    counts = np.bincount(nearest, minlength=len(refs)) / len(pix)
    present = {n: float(c) for n, c in zip(COLORS_RGB, counts) if c > 0.05}
    return len(present), present


def diameter(contour, px_per_mm=None):
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    d = np.linalg.norm(hull[:, None] - hull[None], axis=2)
    i, j = np.unravel_index(d.argmax(), d.shape)
    px = float(d[i, j])
    mm = px / px_per_mm if px_per_mm else None
    # Stolz D: number of dermoscopic structures (approximated here via size/texture)
    d_score = 5 if (mm and mm > 6) else (3 if mm is None and px > 200 else 1)
    return d_score, px, mm, (tuple(hull[i].astype(int)), tuple(hull[j].astype(int)))


# ---------------------------------------------------------------- report / viz
def analyze(path, px_per_mm=None, out="results", show=True):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = resize(img)
    clean, gray, hair_mask, enh = preprocess(img)
    fov = field_of_view(gray)
    mask, contour, raw_th = segment(enh, fov)
    validate_lesion(clean, gray, mask, contour, fov)

    A, a_axes, rot = asymmetry(mask, contour)
    B, bstats = border(contour)
    C, colors = color(clean, mask)
    D, d_px, d_mm, (p1, p2) = diameter(contour, px_per_mm)
    tds = 1.3 * A + 0.1 * B + 0.5 * C + 0.5 * D
    verdict = "Benign-looking" if tds < 4.75 else ("Suspicious" if tds <= 5.45 else "Highly suspicious")

    # overlay
    ov = img.copy()
    cv2.drawContours(ov, [contour], -1, (0, 255, 0), 2)
    cv2.drawContours(ov, [cv2.convexHull(contour)], -1, (255, 200, 0), 1)
    cv2.line(ov, p1, p2, (0, 0, 255), 2)
    if len(contour) >= 5:
        ell = cv2.fitEllipse(contour)
        cv2.ellipse(ov, ell, (255, 0, 255), 1)
        (cx, cy), (MA, ma), ang = ell
        for t in (ang, ang + 90):  # symmetry axes
            r = np.deg2rad(t)
            dx, dy = np.sin(r) * max(MA, ma) / 2, -np.cos(r) * max(MA, ma) / 2
            cv2.line(ov, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), (0, 255, 255), 1)

    # symmetry visual: original (green) vs. flipped (red)
    sym = cv2.merge([np.zeros_like(rot), rot, cv2.flip(rot, 1)])

    lines = [
        f"A asym: {A}  (axes {a_axes[0]:.2f}, {a_axes[1]:.2f})",
        f"B border: {B}  compact {bstats['compactness']:.2f} solid {bstats['solidity']:.2f}",
        f"C colors: {C}  {', '.join(colors)}",
        f"D diam: {d_px:.0f}px" + (f" / {d_mm:.1f}mm" if d_mm else ""),
        f"TDS = {tds:.2f}  -> {verdict}",
        "NOT A MEDICAL DIAGNOSIS",
    ]
    h, w = img.shape[:2]
    panel = np.full((h, w, 3), 30, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if i == 4 else (255, 255, 255), 1, cv2.LINE_AA)

    def lab(im, t):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im.copy()
        cv2.putText(im, t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return im

    grid = np.vstack([
        np.hstack([lab(img, "1 original"), lab(gray, "2 grayscale"), lab(hair_mask, "3 hair mask"), lab(enh, "4 CLAHE")]),
        np.hstack([lab(raw_th, "5 Otsu"), lab(mask, "6 lesion mask"), lab(ov, "7 contour/axes"), lab(sym, "8 symmetry")]),
    ])
    panel_row = np.hstack([panel, np.zeros((h, w * 3, 3), np.uint8)])
    grid = np.vstack([grid, panel_row])

    os.makedirs(out, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(out, f"{base}_steps.png"), grid)
    cv2.imwrite(os.path.join(out, f"{base}_overlay.png"), ov)
    print("\n".join(lines))
    print(f"Saved -> {out}/{base}_steps.png")

    if show:
        disp = resize(grid, 1400)
        cv2.imshow("Melanoma Busters", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return dict(A=A, B=B, C=C, D=D, TDS=tds, verdict=verdict, colors=colors,
                diameter_px=d_px, diameter_mm=d_mm, **bstats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--px-per-mm", type=float, default=None, help="scale for mm diameter")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    try:
        analyze(a.image, a.px_per_mm, a.out, not a.no_show)
    except NoLesionError as e:
        print("MOLE NOT DETECTED")
        for r in e.reasons:
            print("  -", r)
        raise SystemExit(2)
