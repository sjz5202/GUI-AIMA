import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont

def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.array(x)
def _to_pixel_points(points, W, H):
    px = []
    for (x, y) in points:
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            px.append((int(round(x * W)), int(round(y * H))))
        else:
            px.append((int(round(x)), int(round(y))))
    return px
def _to_pixel_box(box, W, H, normalized=True):
    """
    box: (x1, y1, x2, y2)
    normalized=True represent the input is [0,1] coordinates; False represent the input is already in pixels
    return the integer pixel coordinates (x1, y1, x2, y2) and do boundary clipping
    """
    x1, y1, x2, y2 = box
    if normalized:
        x1 = int(round(x1 * W)); y1 = int(round(y1 * H))
        x2 = int(round(x2 * W)); y2 = int(round(y2 * H))
    else:
        x1 = int(round(x1)); y1 = int(round(y1))
        x2 = int(round(x2)); y2 = int(round(y2))
    # normalize and do boundary clipping
    x1, x2 = max(0, min(W-1, x1)), max(0, min(W-1, x2))
    y1, y2 = max(0, min(H-1, y1)), max(0, min(H-1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2
def _normalize_map(m):
    m = m.astype(np.float32)
    m -= m.min()
    denom = (m.max() - m.min())
    if denom < 1e-12:
        return np.zeros_like(m)
    return m / denom

def _reshape_attn(attn_scores, patch_w, patch_h):
    """
    accept multiple shapes:
      - [N]                     -> [1, 1, N]
      - [H, N]                  -> [1, H, N]
      - [L, H, N]               -> [L, H, N]
    return (L, H, patch_h, patch_w)
    """
    A = _to_numpy(attn_scores)
    if A.ndim == 1:
        A = A[None, None, :]                 # [1, 1, N]
    elif A.ndim == 2:
        A = A[None, :, :]                    # [1, H, N]
    elif A.ndim == 3:
        pass                                 # [L, H, N]
    else:
        raise ValueError(f"Unexpected attn_scores ndim={A.ndim}, shape={A.shape}")

    L, H, N = A.shape
    assert N == patch_w * patch_h, f"N={N} != patch_w*patch_h={patch_w*patch_h}"
    # reshape to [L, H, patch_h, patch_w] (usually first row then column: y=h, x=w)
    A = A.reshape(L, H, patch_h, patch_w)
    return A

def _upsample_to_image(attn_map, target_w, target_h, mode=Image.BILINEAR):
    """
    attn_map: 2D numpy [patch_h, patch_w] in [0,1]
    return the PIL grayscale image (0-255) with the same size as the image
    """
    heat = (attn_map * 255.0).astype(np.uint8)
    heat_img = Image.fromarray(heat, mode="L")  # 单通道
    heat_img = heat_img.resize((target_w, target_h), resample=mode)
    return heat_img

def _apply_colormap(gray_img, cmap_name="turbo", alpha=0.45):
    """
    gray_img: PIL L (0-255) image
    return the RGBA image with alpha channel
    """
    arr = np.array(gray_img, dtype=np.float32) / 255.0
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(arr)  # RGBA in [0,1]
    colored[..., 3] = alpha
    colored_img = Image.fromarray((colored * 255).astype(np.uint8), mode="RGBA")
    return colored_img

def _draw_grid(img, patch_w, patch_h, color=(255, 255, 255, 120), width=1):
    """draw the patch grid on the RGBA image (optional)"""
    W, H = img.size
    gx = W / patch_w
    gy = H / patch_h
    draw = ImageDraw.Draw(img)
    # vertical lines
    for i in range(1, patch_w):
        x = int(round(i * gx))
        draw.line([(x, 0), (x, H)], fill=color, width=width)
    # horizontal lines
    for j in range(1, patch_h):
        y = int(round(j * gy))
        draw.line([(0, y), (W, y)], fill=color, width=width)
    return img

def overlay_attention(
    image_pil,
    attn_scores,
    patch_w, patch_h,
    merge="mean",
    cmap="turbo",
    alpha=0.45,
    draw_grid=False,
    grid_width=1,
    mark_points=None,
    # BBox related
    gt_bboxes=None, gt_normalized=True, gt_color=(0,255,0,255), gt_width=3,
    pred_bboxes=None, pred_normalized=True, pred_color=(255,255,0,255), pred_width=3,
    draw_bbox_label=True
):
    # -------- first define the base image --------
    base = image_pil.convert("RGBA")
    W, H = base.size

    # ---- calculate the heatmap ----
    A = _reshape_attn(attn_scores, patch_w, patch_h)  # [L,H,ph,pw]
    L, Hh, ph, pw = A.shape

    A_norm = np.empty_like(A, dtype=np.float32)
    for l in range(L):
        for h in range(Hh):
            A_norm[l, h] = _normalize_map(A[l, h])

    if merge == "mean":
        M = A_norm.mean(axis=(0,1))  # [ph,pw]
    elif isinstance(merge, tuple) and merge[0] == "weighted":
        weights = np.array(merge[1], dtype=np.float32)  # [L,H]
        assert weights.shape == (L, Hh)
        w = weights / (weights.sum() + 1e-12)
        M = (A_norm * w[:, :, None, None]).sum(axis=(0,1))
    else:
        raise ValueError("merge must be 'mean' or ('weighted', weights)")

    # upscale and color (ensure RGBA & size consistent)
    gray = _upsample_to_image(M, W, H, mode=Image.BILINEAR)    # PIL 'L'
    color_overlay = _apply_colormap(gray, cmap_name=cmap, alpha=alpha)  # PIL 'RGBA'
    if color_overlay.size != (W, H):
        color_overlay = color_overlay.resize((W, H), resample=Image.BILINEAR)
    if color_overlay.mode != "RGBA":
        color_overlay = color_overlay.convert("RGBA")

    # composite
    out = Image.alpha_composite(base, color_overlay)

    # ---- draw boxes ----
    def _to_pixel_box(box, W, H, normalized=True):
        x1, y1, x2, y2 = box
        if normalized:
            x1 = int(round(x1 * W)); y1 = int(round(y1 * H))
            x2 = int(round(x2 * W)); y2 = int(round(y2 * H))
        else:
            x1 = int(round(x1)); y1 = int(round(y1))
            x2 = int(round(x2)); y2 = int(round(y2))
        # do boundary clipping and correct the order
        x1, x2 = max(0, min(W-1, x1)), max(0, min(W-1, x2))
        y1, y2 = max(0, min(H-1, y1)), max(0, min(H-1, y2))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return x1, y1, x2, y2

    def _draw_boxes(draw, bboxes, normalized, color, width, tag, draw_bbox_label=False):
        if bboxes is None:
            return
        if isinstance(bboxes, tuple) and len(bboxes) == 4:
            bboxes = [bboxes]
        for box in bboxes:
            x1, y1, x2, y2 = _to_pixel_box(box, W, H, normalized=normalized)
            # only draw the outline, no fill (transparent)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

            # draw the text directly on the image, no background; use the outline to ensure readability
            if draw_bbox_label:
                draw.text(
                    (x1 + 4, y1 + 4),
                    tag,
                    fill=color,                 # same color as the box
                    stroke_width=2,             # outline
                    stroke_fill=(0, 0, 0, 255)  # black outline, avoid mixing with the base image
                )

    draw = ImageDraw.Draw(out)
    _draw_boxes(draw, gt_bboxes,   gt_normalized,   gt_color,   gt_width,   "GT")
    # _draw_boxes(draw, pred_bboxes, pred_normalized, pred_color, pred_width, "Pred")

    # ---- draw the grid ----
    if draw_grid:
        out = _draw_grid(out, patch_w, patch_h, width=grid_width)

    # ---- draw the points (support normalized/pixel)----
    if mark_points:
        r = max(3, min(W, H)//200)
        if isinstance(mark_points, list):
            for points in mark_points:
                points=[points]
                for (x, y) in points:
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                        x = int(round(x * W)); y = int(round(y * H))
                    else:
                        x = int(round(x)); y = int(round(y))
                    # draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=(0, 0, 0, 255), width=2)
                    draw.line([(x - 2*r, y), (x + 2*r, y)], fill=(0, 0, 0, 255), width=2)
                    draw.line([(x, y - 2*r), (x, y + 2*r)], fill=(0, 0, 0, 255), width=2)
        else:
            (x, y) = mark_points
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                x = int(round(x * W)); y = int(round(y * H))
            else:
                x = int(round(x)); y = int(round(y))
            # draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=(0, 0, 0, 255), width=2)
            draw.line([(x - 2*r, y), (x + 2*r, y)], fill=(0, 0, 0, 255), width=2)
            draw.line([(x, y - 2*r), (x, y + 2*r)], fill=(0, 0, 0, 255), width=2)

    return out

def save_headwise_panels(image_pil, A, patch_w, patch_h, out_png, cmap="turbo", alpha=0.45):
    """
    save the heatmap of each head as a horizontal mosaic, for quick comparison.
    A: shape [L,H,N] or [H,N] or [N], internal will reshape
    """
    base = image_pil.convert("RGBA")
    W, H = base.size
    A4 = _reshape_attn(A, patch_w, patch_h)    # [L,H,ph,pw]
    L, Hh, _, _ = A4.shape

    tiles = []
    for l in range(L):
        for h in range(Hh):
            m = _normalize_map(A4[l, h])
            gray = _upsample_to_image(m, W, H)
            overlay = _apply_colormap(gray, cmap_name=cmap, alpha=alpha)
            tile = Image.alpha_composite(base, overlay)
            # draw the label in the corner (optional)
            draw = ImageDraw.Draw(tile)
            draw.rectangle([5, 5, 120, 36], fill=(0,0,0,120))
            draw.text((10, 10), f"L{l}H{h}", fill=(255,255,255,255))
            tiles.append(tile)

    # concatenate
    cols = min(Hh * L, 6)  # at most 6 tiles per row
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGBA", (cols * W, rows * H), (0,0,0,0))
    for idx, im in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        canvas.paste(im, (c * W, r * H))
    canvas.save(out_png)

def crop_subimage(img_width, img_height, px, py, subimage_size):
    """
    Calculates the top-left (x, y) and bottom-right (x, y) coordinates for a subimage crop
    of a fixed size, centered around a given point (px, py).
    Handles boundary conditions by shifting the crop window to ensure it stays within
    the image dimensions while maintaining the subimage_size.

    Args:
        img_width (int): The width of the original image.
        img_height (int): The height of the original image.
        px (int): The x-coordinate of the center point for the crop.
        py (int): The y-coordinate of the center point for the crop.
        subimage_size (int): The desired width and height of the square subimage.

    Returns:
        tuple: A tuple (start_x, start_y, end_x, end_y) representing the
               bounding box coordinates for the subimage.
    """
    half_subimage_size = subimage_size // 2

    # Calculate potential start and end coordinates for x
    start_x_potential = px - half_subimage_size
    end_x_potential = px + half_subimage_size

    # Adjust x-coordinates for boundaries
    if start_x_potential < 0:
        start_x = 0
        end_x = subimage_size
    elif end_x_potential > img_width:
        end_x = img_width
        start_x = img_width - subimage_size
    else:
        start_x = start_x_potential
        end_x = end_x_potential

    # Calculate potential start and end coordinates for y
    start_y_potential = py - half_subimage_size
    end_y_potential = py + half_subimage_size

    # Adjust y-coordinates for boundaries
    if start_y_potential < 0:
        start_y = 0
        end_y = subimage_size
    elif end_y_potential > img_height:
        end_y = img_height
        start_y = img_height - subimage_size
    else:
        start_y = start_y_potential
        end_y = end_y_potential

    # Ensure all coordinates are integers
    return int(start_x), int(start_y), int(end_x), int(end_y)

def create_overlay_image(image, attn_scores, patch_w, patch_h, topk_points, 
                         norm_px, norm_py, norm_x1, norm_y1, norm_x2, norm_y2, 
                         instruction_text):
    """
    Creates an overlay image with attention visualization, predicted point, 
    ground truth bounding box, and instruction text.
    
    Args:
        image: PIL Image object
        attn_scores: Attention scores for overlay
        patch_w: Patch width
        patch_h: Patch height
        topk_points: List of top-k predicted points (normalized coordinates)
        norm_px, norm_py: Normalized predicted point coordinates (top-1)
        norm_x1, norm_y1, norm_x2, norm_y2: Normalized ground truth bbox coordinates
        instruction_text: Text instruction to display on image
        
    Returns:
        PIL Image with overlay visualization
    """
    vis_img = image
    overlay_img = overlay_attention(
        vis_img, attn_scores, patch_w, patch_h,
        merge="mean", cmap="turbo", alpha=0.45,
        draw_grid=True, grid_width=1,
        mark_points=topk_points
    )

    # Ensure overlay_img is in RGB mode for drawing colors and text
    if overlay_img.mode not in ('RGB', 'RGBA'):
        overlay_img = overlay_img.convert('RGB')

    img_width, img_height = overlay_img.size

    # Convert normalized coordinates to pixel coordinates
    px = int(norm_px * img_width)
    py = int(norm_py * img_height)
    x1 = int(norm_x1 * img_width)
    y1 = int(norm_y1 * img_height)
    x2 = int(norm_x2 * img_width)
    y2 = int(norm_y2 * img_height)

    draw = ImageDraw.Draw(overlay_img)

    # Draw the predicted point (px, py)
    point_radius = 8
    draw.ellipse(
        (px - point_radius, py - point_radius, px + point_radius, py + point_radius),
        fill="red",
        outline="red"
    )

    # Draw the ground truth bounding box (gt_bbox)
    bbox_line_width = 3
    draw.rectangle(
        [x1, y1, x2, y2],
        outline="blue",
        width=bbox_line_width
    )

    # Define font and size
    try:
        # Try to load a common font; adjust path or font name if needed
        # For macOS, 'Arial.ttf' or 'Helvetica.ttf' are usually available.
        # For Linux, 'DejaVuSans.ttf' or 'FreeSans.ttf' are common.
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text size using textbbox
    # The bbox format is (left, top, right, bottom)
    if instruction_text is not None:
        _, _, text_w, text_h = draw.textbbox((0, 0), instruction_text, font=font)

        # Calculate position for bottom-right:
        # Add some padding from the bottom and right edges
        padding = 10
        text_x = img_width - text_w - padding
        text_y = img_height - text_h - padding
        
        # Draw the text
        # Using 'white' color for text and a 'black' stroke for better visibility
        draw.text((text_x, text_y), instruction_text, font=font, fill="white", 
                stroke_fill="black", stroke_width=2)
    
    return overlay_img