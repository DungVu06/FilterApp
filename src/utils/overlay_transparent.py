import cv2
import numpy as np

def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]
    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h, :]
    if x < 0:
        w = w + x
        overlay = overlay[:, -x:]
        x = 0
    if y < 0:
        h = h + y
        overlay = overlay[-y:, :]
        y = 0

    if w <= 0 or h <= 0:
        return background

    overlay_image = overlay[..., :3]  # BGR
    mask = overlay[..., 3:] / 255.0  # Alpha

    background_section = background[y:y+h, x:x+w]

    # --- CÔNG THỨC TOÁN HỌC TRỘN ALPHA ---
    # Ảnh mới = (Ảnh Filter * Độ trong suốt filter) + (Ảnh nền * (1 - Độ trong suốt filter))
    composite = (overlay_image * mask) + (background_section * (1.0 - mask))

    background[y:y+h, x:x+w] = composite.astype(np.uint8)

    return background