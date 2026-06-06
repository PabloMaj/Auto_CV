import subprocess
import sys
from pathlib import Path
from string import Template

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Demo app template  (string.Template — $ substitutions, no {braces} issues)
# ---------------------------------------------------------------------------

_DEMO_TEMPLATE = Template(
    '''\
import tkinter as tk
from tkinter import filedialog
import threading
import os
import cv2
import numpy as np
from PIL import Image, ImageTk

# ===== EMBEDDED PREDICTOR =====
$predictor_code
# ===== END OF PREDICTOR =====

USER_PROMPT = """$user_prompt"""
OUTPUT_TYPE = "$desired_output"


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _draw_predictions(img_bgr, predictions, output_type):
    vis = img_bgr.copy()
    palette = [
        (0, 220, 100), (30, 140, 255), (255, 80,  40),
        (200, 50, 200), (0, 200, 220), (255, 200, 0),
    ]
    for i, pred in enumerate(predictions):
        color = palette[i % len(palette)]
        if output_type == "line_segments":
            seg = pred.get("line") or pred.get("line_segment")
            if seg and len(seg) == 4:
                cv2.line(vis, (int(seg[0]), int(seg[1])),
                         (int(seg[2]), int(seg[3])), color, 2)
        elif output_type == "bounding_boxes":
            bbox = pred.get("bbox") or pred.get("bounding_box")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        elif output_type in ("points", "midpoints"):
            pt = pred.get("point") or pred.get("midpoint")
            if pt and len(pt) >= 2:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, (255, 255, 255), 1)
        else:
            # generic: try all known keys
            for key in ("line", "bbox", "point", "midpoint"):
                val = pred.get(key)
                if val:
                    if len(val) == 4:
                        cv2.line(vis, (int(val[0]), int(val[1])),
                                 (int(val[2]), int(val[3])), color, 2)
                    elif len(val) == 2:
                        cv2.circle(vis, (int(val[0]), int(val[1])), 6, color, -1)
                    break
    return vis


def _fit_to_label(img_bgr, label):
    label.update_idletasks()
    max_w = max(label.winfo_width(), 460)
    max_h = max(label.winfo_height(), 360)
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    photo = ImageTk.PhotoImage(pil_img)
    label.config(image=photo)
    label._photo = photo  # keep reference


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

BG       = "#1e1e2e"
SURFACE  = "#313244"
OVERLAY  = "#45475a"
TEXT     = "#cdd6f4"
SUBTEXT  = "#6c7086"
BLUE     = "#89b4fa"
GREEN    = "#a6e3a1"
YELLOW   = "#f9e2af"
RED      = "#f38ba8"
FONT_CODE = ("Consolas", 10)
FONT_BOLD = ("Consolas", 10, "bold")
FONT_TINY = ("Consolas", 8)


class DemoApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CV Prediction Demo")
        self.root.geometry("1160x760")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self._predictor = Predictor()
        self._img_path: str = ""
        self._orig_img_bgr = None
        self._res_img_bgr  = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── header: user prompt ──────────────────────────────────────
        hdr = tk.Frame(self.root, bg=SURFACE, padx=12, pady=8)
        hdr.pack(fill=tk.X, padx=10, pady=(10, 0))

        tk.Label(hdr, text="User Prompt", font=FONT_BOLD,
                 fg=BLUE, bg=SURFACE).pack(anchor="w")

        prompt_box = tk.Text(
            hdr, height=3, font=FONT_CODE,
            fg=TEXT, bg=OVERLAY, relief="flat",
            wrap=tk.WORD, state="normal",
            borderwidth=0, highlightthickness=0,
        )
        prompt_box.insert("1.0", USER_PROMPT)
        prompt_box.config(state="disabled")
        prompt_box.pack(fill=tk.X, pady=(4, 0))

        # ── controls ─────────────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg=BG, pady=8)
        ctrl.pack(fill=tk.X, padx=10)

        self._browse_btn = tk.Button(
            ctrl, text="  Browse Image  ", command=self._browse,
            font=FONT_BOLD, fg=BG, bg=BLUE,
            relief="flat", padx=10, pady=5, cursor="hand2",
        )
        self._browse_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._run_btn = tk.Button(
            ctrl, text="  Run Prediction  ", command=self._run_prediction,
            font=FONT_BOLD, fg=BG, bg=GREEN,
            relief="flat", padx=10, pady=5, cursor="hand2",
            state="disabled",
        )
        self._run_btn.pack(side=tk.LEFT)

        self._status = tk.Label(
            ctrl, text="No image loaded",
            font=FONT_CODE, fg=SUBTEXT, bg=BG,
        )
        self._status.pack(side=tk.LEFT, padx=14)

        # output type badge
        tk.Label(
            ctrl, text=f"output: {OUTPUT_TYPE}",
            font=FONT_TINY, fg=SUBTEXT, bg=BG,
        ).pack(side=tk.RIGHT)

        # ── image panels ─────────────────────────────────────────────
        panels = tk.Frame(self.root, bg=BG)
        panels.pack(fill=tk.BOTH, expand=True, padx=10)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(0, weight=1)

        self._orig_lbl  = self._make_image_panel(panels, "Original",          BLUE,  0)
        self._res_lbl   = self._make_image_panel(panels, "Prediction Result",  GREEN, 1)

        self._orig_lbl.bind("<Button-1>", lambda e: self._open_zoom(self._orig_img_bgr))
        self._res_lbl.bind("<Button-1>",  lambda e: self._open_zoom(self._res_img_bgr))
        self._orig_lbl.config(cursor="hand2")
        self._res_lbl.config(cursor="hand2")

        # ── log ───────────────────────────────────────────────────────
        log_frame = tk.Frame(self.root, bg=BG)
        log_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        self._log = tk.Text(
            log_frame, height=4, font=FONT_TINY,
            fg=TEXT, bg="#181825", relief="flat",
            state="disabled", borderwidth=0, highlightthickness=0,
        )
        self._log.pack(fill=tk.X)

    def _make_image_panel(self, parent, title, title_color, col):
        frame = tk.Frame(parent, bg=SURFACE, padx=4, pady=4)
        frame.grid(row=0, column=col,
                   sticky="nsew", padx=(0 if col == 0 else 4, 0))

        tk.Label(frame, text=title, font=FONT_BOLD,
                 fg=title_color, bg=SURFACE).pack(anchor="w")

        lbl = tk.Label(frame, bg=OVERLAY, relief="flat",
                       text="—", fg=SUBTEXT, font=FONT_CODE)
        lbl.pack(fill=tk.BOTH, expand=True)
        return lbl

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._img_path = path
        img = cv2.imread(path)
        if img is None:
            self._set_status("Cannot read image", RED)
            return
        self._orig_img_bgr = img
        self._res_img_bgr  = None
        _fit_to_label(img, self._orig_lbl)
        self._res_lbl.config(image="", text="—")
        self._run_btn.config(state="normal")
        self._set_status(os.path.basename(path), TEXT)
        self._write_log(f"Loaded: {path}")

    def _run_prediction(self):
        if not self._img_path:
            return
        self._run_btn.config(state="disabled")
        self._browse_btn.config(state="disabled")
        self._set_status("Running prediction…", YELLOW)

        def _task():
            try:
                preds = self._predictor.predict(self._img_path)
                img   = cv2.imread(self._img_path)
                vis   = _draw_predictions(img, preds, OUTPUT_TYPE)
                self.root.after(0, lambda: self._on_done(vis, preds))
            except Exception as exc:
                self.root.after(0, lambda: self._on_error(str(exc)))

        threading.Thread(target=_task, daemon=True).start()

    def _on_done(self, vis_img, preds):
        self._res_img_bgr = vis_img
        _fit_to_label(vis_img, self._res_lbl)
        self._set_status(f"Done — {len(preds)} detection(s)", GREEN)
        self._run_btn.config(state="normal")
        self._browse_btn.config(state="normal")
        self._write_log(f"Prediction complete: {len(preds)} detection(s)")

    def _on_error(self, msg):
        self._set_status("Error — see log", RED)
        self._run_btn.config(state="normal")
        self._browse_btn.config(state="normal")
        self._write_log(f"ERROR: {msg}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open_zoom(self, img_bgr):
        if img_bgr is None:
            return
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        max_w, max_h = int(sw * 0.9), int(sh * 0.9)

        h, w = img_bgr.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        win = tk.Toplevel(self.root)
        win.title("Zoom")
        win.configure(bg=BG)
        win.geometry(f"{new_w}x{new_h}")
        win.resizable(False, False)
        win.bind("<Escape>", lambda e: win.destroy())
        win.bind("<Button-1>", lambda e: win.destroy())

        lbl = tk.Label(win, image=photo, bg=BG, cursor="hand2")
        lbl._photo = photo
        lbl.pack()

    def _set_status(self, text, color=TEXT):
        self._status.config(text=text, fg=color)

    def _write_log(self, msg):
        self._log.config(state="normal")
        self._log.insert(tk.END, f"> {msg}\\n")
        self._log.see(tk.END)
        self._log.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()
'''
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DemoBuilderAgent:

    def __init__(self):
        self.repo_root = Path(__file__).resolve().parents[2]

    def run(self, state):
        logger.info("Running DemoBuilderAgent")

        exp_id = state.get("exp_id", "default")
        self.workspace_dir = (self.repo_root / "workspace" / exp_id).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        user_prompt = state.get("user_prompt", "")
        desired_output = state.get("desired_output", "unknown")

        code, source = self._load_best_code_by_metric()
        if not code:
            # fallback: use in-memory code from state
            code = state.get("generated_code", "")
            source = "state (no metrics found in workspace)"

        if not code.strip():
            logger.warning("DemoBuilderAgent: no code available — skipping")
            return state

        logger.info(f"Using code from: {source}")

        demo_py = self.workspace_dir / "demo_app.py"
        demo_py.write_text(
            _DEMO_TEMPLATE.safe_substitute(
                predictor_code=self._strip_main_block(code),
                user_prompt=user_prompt.replace("\\", "\\\\").replace('"', '\\"'),
                desired_output=desired_output,
            ),
            encoding="utf-8",
        )
        logger.info(f"Demo script written: {demo_py}")

        exe_path = self._build_exe(demo_py)
        if exe_path:
            state["demo_app_exe_path"] = str(exe_path)
            logger.info(f"Demo exe ready: {exe_path}")

        return state

    # ------------------------------------------------------------------

    def _load_best_code_by_metric(self):
        """Scan all stage_X_step_X dirs, pick the one with highest metric_value."""
        import json
        import re

        best_value = float("-inf")
        best_code = ""
        best_source = ""

        pattern = re.compile(r"^stage_(\d+)_step_(\d+)$")

        for folder in self.workspace_dir.iterdir():
            if not folder.is_dir() or not pattern.match(folder.name):
                continue

            metrics_path = folder / "evaluation" / "metrics" / "val" / "val_metrics.json"
            solution_path = folder / "generated_solution.py"

            if not metrics_path.exists() or not solution_path.exists():
                continue

            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                value = float(data.get("metric_value", float("-inf")))
            except Exception as exc:
                logger.warning(f"Could not read metrics from {metrics_path}: {exc}")
                continue

            logger.info(f"  {folder.name}: metric_value={value}")

            if value > best_value:
                best_value = value
                best_code = solution_path.read_text(encoding="utf-8")
                best_source = f"{folder.name} (metric_value={value})"

        return best_code, best_source

    def _strip_main_block(self, code: str) -> str:
        idx = code.find("\nif __name__")
        return code[:idx] if idx != -1 else code

    def _build_exe(self, script_path: Path):
        build_dir = self.workspace_dir / "_pyinstaller_build"
        build_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--windowed",
            "--name", "demo_app",
            "--distpath", str(self.workspace_dir),
            "--workpath", str(build_dir),
            "--specpath", str(build_dir),
            "--hidden-import", "PIL._tkinter_finder",
            str(script_path),
        ]

        logger.info("Building exe with PyInstaller (this may take a few minutes)…")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                exe = self.workspace_dir / "demo_app.exe"
                if exe.exists():
                    return exe
                logger.error("PyInstaller succeeded but demo_app.exe not found")
            else:
                logger.error(f"PyInstaller failed:\n{result.stderr[-3000:]}")
        except subprocess.TimeoutExpired:
            logger.error("PyInstaller timed out after 10 minutes")
        except Exception as exc:
            logger.error(f"PyInstaller error: {exc}")
        return None
