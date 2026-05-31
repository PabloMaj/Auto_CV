def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a["bbox"] if isinstance(a, dict) else a
    bx1, by1, bx2, by2 = b["bbox"] if isinstance(b, dict) else b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    return inter / (area_a + area_b - inter + 1e-9)
