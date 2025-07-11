import cv2
import numpy as np
import pyautogui
import keyboard
import os
import time
import json
from PIL import Image, ImageTk
import tkinter as tk
import screeninfo

# --- CONFIGURABLE PARAMETERS ---
# Each cutout region has unique coordinates (x1, y1, x2, y2)
# Adjust these for your setup (values are EXAMPLES)
CUTOUT_REGIONS = [
    (473, 268, 594, 391),
    (617, 270, 739, 393),
    (474, 412, 595, 533),
    (619, 414, 739, 533),
    (472, 558, 594, 679),
    (618, 559, 738, 678),
    (475, 702, 597, 822),
    (617, 703, 738, 820),
]
MATCH_THRESHOLD = 3e8

# Main print region (x, y, w, h) - adjust as needed
MAIN_PRINT_REGION = (800, 200, 200, 290)

# --- LOAD FINGERPRINT TEMPLATES ---
dir_path = os.path.dirname(os.path.realpath(__file__))
prints = [cv2.imread(os.path.join(dir_path, f"print_{i}.png")) for i in range(4)]

REGION_FILE = 'fingerprint_regions.json'

# --- Default regions (used if no file exists) ---
default_main_print_region = [800, 200, 1000, 490]  # x1, y1, x2, y2 (x, y, x+w, y+h)
default_cutout_regions = [
    [473, 268, 594, 391],
    [617, 270, 739, 393],
    [474, 412, 595, 533],
    [619, 414, 739, 533],
    [472, 558, 594, 679],
    [618, 559, 738, 678],
    [475, 702, 597, 822],
    [617, 703, 738, 820],
]

# --- Load or set regions ---
def load_regions():
    if os.path.exists(REGION_FILE):
        with open(REGION_FILE, 'r') as f:
            data = json.load(f)
        return data['main_print_region'], data['cutout_regions']
    else:
        return default_main_print_region, default_cutout_regions

def save_regions(main_print_region, cutout_regions):
    with open(REGION_FILE, 'w') as f:
        json.dump({
            'main_print_region': main_print_region,
            'cutout_regions': cutout_regions
        }, f, indent=2)


def get_centered_16_9_crop(img):
    h, w = img.shape[:2]
    # Calculate 16:9 crop
    target_w = w
    target_h = int(w * 9 / 16)
    if target_h > h:
        target_h = h
        target_w = int(h * 16 / 9)
    x1 = (w - target_w) // 2
    y1 = (h - target_h) // 2
    x2 = x1 + target_w
    y2 = y1 + target_h
    return img[y1:y2, x1:x2].copy(), x1, y1


# --- Interactive region setup window ---
def show_region_setup(img, window_name="Setup Fingerprint Regions"):
    from PIL import Image, ImageTk
    import tkinter as tk
    import screeninfo

    # Convert OpenCV image (BGR) to PIL image (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Get monitor info
    monitors = screeninfo.get_monitors()
    if len(monitors) > 1:
        m = monitors[1]
    else:
        m = monitors[0]

    root = tk.Tk()
    root.title(window_name)
    root.attributes('-topmost', True)
    root.geometry(f"{m.width}x{m.height}+{m.x}+{m.y}")
    root.attributes('-fullscreen', True)
    root.focus_force()

    pil_img = pil_img.resize((m.width, m.height))
    tk_img = ImageTk.PhotoImage(pil_img)

    canvas = tk.Canvas(root, width=m.width, height=m.height)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

    # Load or set regions
    main_print_region, cutout_regions = load_regions()
    main_print_rect = list(main_print_region)
    rects = [list(region) for region in cutout_regions]
    selected = None
    drag_offset = (0, 0)
    resizing = False
    resize_corner = None
    handle_size = 10
    selected_type = None  # 'main' or 'cutout'

    def draw_rects():
        canvas.delete("rect")
        # Draw main print region (blue)
        x1, y1, x2, y2 = main_print_rect
        canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=3, tags="rect")
        canvas.create_text(x1+10, y1+20, text="Main", fill="blue", font=("Arial", 16), tags="rect")
        for (hx, hy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            canvas.create_rectangle(hx-handle_size, hy-handle_size, hx+handle_size, hy+handle_size, fill="light blue", tags="rect")
        # Draw cutout regions (magenta)
        for idx, (x1, y1, x2, y2) in enumerate(rects):
            canvas.create_rectangle(x1, y1, x2, y2, outline="magenta", width=2, tags="rect")
            canvas.create_text(x1+10, y1+20, text=str(idx), fill="yellow", font=("Arial", 16), tags="rect")
            for (hx, hy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                canvas.create_rectangle(hx-handle_size, hy-handle_size, hx+handle_size, hy+handle_size, fill="cyan", tags="rect")

    draw_rects()

    def point_in_rect(x, y, rect):
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def point_in_handle(x, y, rect):
        x1, y1, x2, y2 = rect
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for i, (hx, hy) in enumerate(corners):
            if abs(x - hx) <= handle_size and abs(y - hy) <= handle_size:
                return i  # corner index
        return None

    def on_mouse_down(event):
        nonlocal selected, drag_offset, resizing, resize_corner, selected_type
        # Check main print region first
        corner = point_in_handle(event.x, event.y, main_print_rect)
        if corner is not None:
            selected = 0
            selected_type = 'main'
            resizing = True
            resize_corner = corner
            return
        if point_in_rect(event.x, event.y, main_print_rect):
            selected = 0
            selected_type = 'main'
            resizing = False
            drag_offset = (event.x - main_print_rect[0], event.y - main_print_rect[1])
            return
        # Then check cutout regions
        for idx, rect in enumerate(rects):
            corner = point_in_handle(event.x, event.y, rect)
            if corner is not None:
                selected = idx
                selected_type = 'cutout'
                resizing = True
                resize_corner = corner
                return
        for idx, rect in enumerate(rects):
            if point_in_rect(event.x, event.y, rect):
                selected = idx
                selected_type = 'cutout'
                resizing = False
                drag_offset = (event.x - rect[0], event.y - rect[1])
                return
        selected = None
        selected_type = None

    def on_mouse_move(event):
        if selected is not None:
            if selected_type == 'main':
                if resizing:
                    x1, y1, x2, y2 = main_print_rect
                    if resize_corner == 0:  # top-left
                        main_print_rect[0] = min(event.x, x2-10)
                        main_print_rect[1] = min(event.y, y2-10)
                    elif resize_corner == 1:  # top-right
                        main_print_rect[2] = max(event.x, x1+10)
                        main_print_rect[1] = min(event.y, y2-10)
                    elif resize_corner == 2:  # bottom-left
                        main_print_rect[0] = min(event.x, x2-10)
                        main_print_rect[3] = max(event.y, y1+10)
                    elif resize_corner == 3:  # bottom-right
                        main_print_rect[2] = max(event.x, x1+10)
                        main_print_rect[3] = max(event.y, y1+10)
                else:
                    x1, y1, x2, y2 = main_print_rect
                    w, h = x2 - x1, y2 - y1
                    new_x1 = event.x - drag_offset[0]
                    new_y1 = event.y - drag_offset[1]
                    main_print_rect[0] = new_x1
                    main_print_rect[1] = new_y1
                    main_print_rect[2] = new_x1 + w
                    main_print_rect[3] = new_y1 + h
            elif selected_type == 'cutout':
                if resizing:
                    x1, y1, x2, y2 = rects[selected]
                    if resize_corner == 0:  # top-left
                        rects[selected][0] = min(event.x, x2-10)
                        rects[selected][1] = min(event.y, y2-10)
                    elif resize_corner == 1:  # top-right
                        rects[selected][2] = max(event.x, x1+10)
                        rects[selected][1] = min(event.y, y2-10)
                    elif resize_corner == 2:  # bottom-left
                        rects[selected][0] = min(event.x, x2-10)
                        rects[selected][3] = max(event.y, y1+10)
                    elif resize_corner == 3:  # bottom-right
                        rects[selected][2] = max(event.x, x1+10)
                        rects[selected][3] = max(event.y, y1+10)
                else:
                    x1, y1, x2, y2 = rects[selected]
                    w, h = x2 - x1, y2 - y1
                    new_x1 = event.x - drag_offset[0]
                    new_y1 = event.y - drag_offset[1]
                    rects[selected][0] = new_x1
                    rects[selected][1] = new_y1
                    rects[selected][2] = new_x1 + w
                    rects[selected][3] = new_y1 + h
            draw_rects()

    def on_mouse_up(event):
        nonlocal selected, resizing, resize_corner, selected_type
        selected = None
        resizing = False
        resize_corner = None
        selected_type = None

    def on_key(event):
        if event.char == 's':
            print('Regions saved to', REGION_FILE)
            save_regions([int(x) for x in main_print_rect], [[int(x) for x in r] for r in rects])
            root.destroy()
        elif event.keysym == 'Escape':
            root.destroy()

    canvas.bind("<ButtonPress-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_move)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)
    root.bind("<Key>", on_key)

    root.mainloop()


# --- Detection logic ---
def detect_and_show(img, main_print_region, cutout_regions, match_threshold=2e6):
    # 1. Detect which print (0-3) is present in the main region
    x1, y1, x2, y2 = main_print_region
    x, y, w, h = x1, y1, x2-x1, y2-y1
    regionPrint = img[y:y+h, x:x+w].copy()
    regionPrint_gray = cv2.cvtColor(regionPrint, cv2.COLOR_BGR2GRAY)
    regionPrint_bin = cv2.inRange(regionPrint_gray, 20, 255)
    maxPos = 0
    maxVal = float('-inf')
    maxLoc = 0
    print('--- Main print detection ---')
    for i in range(4):
        print_img = cv2.imread(os.path.join(dir_path, f"print_{i}.png"))
        print_gray = cv2.cvtColor(print_img, cv2.COLOR_BGR2GRAY)
        print_bin = cv2.inRange(print_gray, 20, 255)
        res = cv2.matchTemplate(regionPrint_bin, print_bin, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(f'  print_{i}: max_val={max_val}')
        if max_val > maxVal:
            maxVal = max_val
            maxLoc = max_loc
            maxPos = i
    detected_print = maxPos
    print(f'Detected print index: {detected_print} (print_{detected_print}.png)')
    # 2. Load the 4 correct cutout images for this print
    correct_cutouts = []
    for idx in range(4):
        p = cv2.imread(os.path.join(dir_path, str(detected_print), f"{idx}.png"))
        p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        p = cv2.inRange(p, 20, 255)
        correct_cutouts.append(p)
    # 3. For each of the 8 cutout regions, match against the 4 correct cutouts
    matches = []  # (region_idx, cutout_idx, max_val)
    used_cutouts = set()
    print('--- Cutout region matching ---')
    for region_idx, (cx1, cy1, cx2, cy2) in enumerate(cutout_regions):
        cutout_img = img[cy1:cy2, cx1:cx2].copy()
        cutout_gray = cv2.cvtColor(cutout_img, cv2.COLOR_BGR2GRAY)
        cutout_bin = cv2.inRange(cutout_gray, 20, 255)
        best_val = float('-inf')
        best_cutout = -1
        for sol_idx, sol_img in enumerate(correct_cutouts):
            ch, cw = cutout_bin.shape[:2]
            th, tw = sol_img.shape[:2]
            print(f'  region {region_idx} vs cutout {sol_idx}: cutout_bin shape={cutout_bin.shape}, sol_img shape={sol_img.shape}')
            if (ch, cw) != (th, tw):
                print(f'    Resizing template from {sol_img.shape} to {cutout_bin.shape}')
                sol_img_resized = cv2.resize(sol_img, (cw, ch), interpolation=cv2.INTER_AREA)
            else:
                sol_img_resized = sol_img
            res = cv2.matchTemplate(cutout_bin, sol_img_resized, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            print(f'    max_val={max_val}')
            if max_val > best_val:
                best_val = max_val
                best_cutout = sol_idx
        matches.append((region_idx, best_cutout, best_val))
    # Sort by match value, pick the best 4 unique cutout matches above threshold
    matches = sorted(matches, key=lambda x: x[2], reverse=True)
    correct_indices = []
    used_cutouts = set()
    for region_idx, cutout_idx, val in matches:
        if val > match_threshold and cutout_idx not in used_cutouts:
            correct_indices.append(region_idx)
            used_cutouts.add(cutout_idx)
            print(f'  Selected: region {region_idx} as correct for cutout {cutout_idx} (val={val})')
        if len(correct_indices) == 4:
            break
    print(f'Final correct region indices: {correct_indices}')
    show_result_window(img, cutout_regions, correct_indices)

# --- Result window (only green squares on correct answers) ---
def show_result_window(img, cutout_regions, correct_indices):
    # Draw only green rectangles on correct cutouts
    img_disp = img.copy()
    for idx in correct_indices:
        cx1, cy1, cx2, cy2 = cutout_regions[idx]
        cv2.rectangle(img_disp, (cx1, cy1), (cx2, cy2), (0,255,0), 4)
    # Show in fullscreen on second monitor
    monitors = screeninfo.get_monitors()
    if len(monitors) > 1:
        m = monitors[1]
    else:
        m = monitors[0]
    window_name = "Fingerprint Result"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, m.x, m.y)
    cv2.resizeWindow(window_name, m.width, m.height)
    cv2.imshow(window_name, img_disp)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


# --- Main logic ---
def main():
    import sys
    setup_mode = '--setup' in sys.argv
    main_print_region, cutout_regions = load_regions()
    if not os.path.exists(REGION_FILE) or setup_mode:
        # Take screenshot and let user adjust regions
        screenshot = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        img_16_9, xoff, yoff = get_centered_16_9_crop(img)
        show_region_setup(img_16_9)
        # After saving, reload regions
        main_print_region, cutout_regions = load_regions()
    while True:
        if keyboard.is_pressed('F10'):
            print("F10 pressed! Capturing screenshot...")
            screenshot = pyautogui.screenshot()
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            img_16_9, xoff, yoff = get_centered_16_9_crop(img)
            detect_and_show(img_16_9, main_print_region, cutout_regions)
            while keyboard.is_pressed('F10'):
                time.sleep(0.1)
        time.sleep(0.05)

if __name__ == "__main__":
    main() 