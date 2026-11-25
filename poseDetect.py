import ctypes
from ctypes import wintypes
import tkinter as tk

import cv2
import mss
import numpy as np
import mediapipe as mp

# make coordinates match real pixels on high-DPI screens
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

user32 = ctypes.windll.user32
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetAncestor.restype = wintypes.HWND
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]

mp_pose = mp.solutions.pose

KEY = "#010101"          # colour that becomes see-through on the overlay
LINE_COLOUR = "#00ff00"
POINT_COLOUR = "#ff3030"
BOX_COLOUR = "#00ffff"
MAX_PROCESS_WIDTH = 640  # frames are shrunk to this width before detection (speed)
MIN_VISIBILITY = 0.5     # hide landmarks the model isn't confident about


def esc_pressed():
    return bool(user32.GetAsyncKeyState(0x1B) & 0x8000)


def make_overlay_window(win):
    """Click-through, always on top, and invisible to screen capture."""
    win.update_idletasks()
    hwnd = user32.GetAncestor(win.winfo_id(), 2)  # GA_ROOT
    GWL_EXSTYLE = -20
    WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW = 0x80000, 0x20, 0x80
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                          style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW)
    # stops the overlay appearing in the screen grab (Windows 10 2004+)
    if not user32.SetWindowDisplayAffinity(hwnd, 0x11):  # WDA_EXCLUDEFROMCAPTURE
        print("Warning: couldn't hide the overlay from capture; "
              "you may see flicker or feedback.")


def select_region(root, virtual):
    """Full-screen dim layer; drag a rectangle; returns a dict for mss or None."""
    vl, vt, vw, vh = virtual["left"], virtual["top"], virtual["width"], virtual["height"]
    chosen = {}

    sel = tk.Toplevel(root)
    sel.overrideredirect(True)
    sel.geometry(f"{vw}x{vh}+{vl}+{vt}")
    sel.attributes("-topmost", True)
    sel.attributes("-alpha", 0.3)
    canvas = tk.Canvas(sel, bg="black", highlightthickness=0, cursor="crosshair")
    canvas.pack(fill="both", expand=True)
    canvas.create_text(vw // 2, 40, fill="white", font=("Segoe UI", 18),
                       text="Drag a box around the area to track  (Esc to cancel)")

    state = {"x0": 0, "y0": 0, "rect": None}

    def on_press(e):
        state["x0"], state["y0"] = e.x_root, e.y_root
        state["rect"] = canvas.create_rectangle(0, 0, 0, 0, outline="red", width=3)

    def on_drag(e):
        if state["rect"] is not None:
            canvas.coords(state["rect"],
                          state["x0"] - vl, state["y0"] - vt,
                          e.x_root - vl, e.y_root - vt)

    def on_release(e):
        x0, y0, x1, y1 = state["x0"], state["y0"], e.x_root, e.y_root
        chosen.update(left=min(x0, x1), top=min(y0, y1),
                      width=abs(x1 - x0), height=abs(y1 - y0))
        sel.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    sel.bind("<Escape>", lambda e: sel.destroy())
    sel.focus_force()

    root.wait_window(sel)
    if chosen.get("width", 0) < 50 or chosen.get("height", 0) < 50:
        return None
    return chosen


def main():
    root = tk.Tk()
    root.withdraw()

    with mss.mss() as sct:
        region = select_region(root, sct.monitors[0])  # monitors[0] = all screens
        if region is None:
            print("No region selected.")
            return

        left, top, w, h = region["left"], region["top"], region["width"], region["height"]

        # transparent, click-through overlay exactly over the chosen region
        overlay = tk.Toplevel(root)
        overlay.overrideredirect(True)
        overlay.geometry(f"{w}x{h}+{left}+{top}")
        overlay.attributes("-topmost", True)
        overlay.attributes("-transparentcolor", KEY)
        canvas = tk.Canvas(overlay, width=w, height=h, bg=KEY, highlightthickness=0)
        canvas.pack()
        canvas.create_rectangle(1, 1, w - 2, h - 2, outline="red", width=2)  # region border
        make_overlay_window(overlay)

        print("Tracking. Press Esc to quit.")

        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:

            def tick():
                if esc_pressed():
                    root.destroy()
                    return

                frame = np.array(sct.grab(region))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                if w > MAX_PROCESS_WIDTH:
                    scale = MAX_PROCESS_WIDTH / w
                    rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)

                results = pose.process(rgb)
                canvas.delete("pose")

                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    pts = [(lm.x * w, lm.y * h, lm.visibility) for lm in lms]

                    for a, b in mp_pose.POSE_CONNECTIONS:
                        if pts[a][2] > MIN_VISIBILITY and pts[b][2] > MIN_VISIBILITY:
                            canvas.create_line(pts[a][0], pts[a][1], pts[b][0], pts[b][1],
                                               fill=LINE_COLOUR, width=3, tags="pose")

                    visible = [p for p in pts if p[2] > MIN_VISIBILITY]
                    for x, y, _ in visible:
                        canvas.create_oval(x - 4, y - 4, x + 4, y + 4,
                                           fill=POINT_COLOUR, outline="", tags="pose")



                root.after(1, tick)

            root.after(0, tick)
            root.mainloop()


if __name__ == "__main__":
    main()