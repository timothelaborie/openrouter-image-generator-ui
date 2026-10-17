"""
OpenRouter Image Studio
-----------------------
A desktop UI for generating / editing images with several OpenRouter image
models in parallel.

Install:   pip install customtkinter pillow requests
Run:       python image_studio.py
API key:   set the OPENROUTER_API_KEY environment variable
           (or type it into the sidebar field; the key is never saved to disk).
"""

import base64
import ctypes
import json
import mimetypes
import os
import queue
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import requests
from PIL import Image, ImageOps

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

IMAGES_URL = "https://openrouter.ai/api/v1/images"
SETTINGS_FILE = Path.home() / ".openrouter_image_studio.json"

MODELS = [
    {"id": "openai/gpt-image-2.5-sunburst",     "name": "GPT Image 2.5 Sunburst", "max_refs": 16},
    {"id": "microsoft/mai-image-2.6",           "name": "MAI-Image-2.6",          "max_refs": 5},
    {"id": "black-forest-labs/flux-3-image",    "name": "FLUX.3 Image",           "max_refs": 10},
    {"id": "bytedance-seed/seedream-5-0-flash", "name": "Seedream 5.0 Flash",     "max_refs": 14},
    {"id": "google/gemini-nano-banana-2.1",     "name": "Nano Banana 2.1",        "max_refs": 14},
    {"id": "qwen/qwen-image-2.1-pro",           "name": "Qwen Image 2.1 Pro",     "max_refs": 4},
    {"id": "tencent/hy-image-v3.5-preview",     "name": "Hy Image 3.5 Preview",   "max_refs": 20},
    {"id": "krea/k3-edit",                      "name": "K3 Edit",                "max_refs": 8},
]

# Aspect ratios shared by these models. K3 Edit rejects 3:4.
DEFAULT_ASPECT_LABEL = "Model default"
ASPECT_RATIOS = [DEFAULT_ASPECT_LABEL, "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"]

MAX_PARALLEL = 16
INPUT_THUMB_BOX = 84
RESULT_THUMB_BOX = 128

DEFAULT_SETTINGS = {
    "prompt": "Remove the phone and make the background blank",
    "folder": "generations",
    "count": 1,
    "aspect": DEFAULT_ASPECT_LABEL,
    "models": {m["id"]: False for m in MODELS},   # all off by default
    "inputs": [],
}

ACCENT = "#6C5CE7"
ACCENT_HOVER = "#5A4BD1"
CARD = "#23262E"
BG = "#181A20"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _primary_work_origin() -> tuple[int, int]:
    """Top-left of the primary monitor's work area, in virtual-screen pixels."""
    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = RECT()
    # SPI_GETWORKAREA is always the primary monitor, taskbar excluded.
    if not ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
        return 0, 0
    return rect.left + 8, rect.top + 8


def model_slug(model_id: str) -> str:
    """'google/gemini-nano-banana-2.1' -> 'google_gemini-nano-banana-2.1'"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def next_image_number(folder: Path, slug: str) -> int:
    """Highest existing '<slug>_<n>.<ext>' in folder + 1 (or 1)."""
    if not folder.exists():
        return 1
    pattern = re.compile(rf"^{re.escape(slug)}_(\d+)\.\w+$")
    numbers = [int(m.group(1)) for f in folder.iterdir() if (m := pattern.match(f.name))]
    return max(numbers) + 1 if numbers else 1


def file_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    if not mime.startswith("image/"):
        mime = "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def load_thumb_pil(path: str, max_side: int):
    """Load a small PIL thumbnail (safe to call from any thread). None on failure."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA")
            im.thumbnail((max_side, max_side))
            return im.copy()
    except Exception:
        return None


def pil_to_ctk(pil_img, box: int) -> ctk.CTkImage:
    w, h = pil_img.size
    scale = min(box / w, box / h)
    return ctk.CTkImage(pil_img, size=(max(1, int(w * scale)), max(1, int(h * scale))))


def open_path(path: str):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        messagebox.showerror("Could not open", str(e))


def ext_from_media_type(media_type: str) -> str:
    known = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp"}
    if media_type in known:
        return known[media_type]
    sub = media_type.split("/")[-1] if "/" in media_type else "png"
    return re.sub(r"[^a-z0-9]", "", sub.lower()) or "png"


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

class ImageStudio(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("OpenRouter Image Studio")
        self.configure(fg_color=BG)
        self.minsize(1050, 740)

        self.settings = self._load_settings()
        self.input_paths: list[str] = [p for p in self.settings["inputs"] if Path(p).exists()]

        self.msg_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.running = False
        self.total_tasks = 0
        self.done_tasks = 0
        self.total_cost = 0.0
        self.ok_count = 0

        self._thumb_cache: dict = {}
        self._result_refs: list = []
        self._result_col = 0

        self.f_title = ctk.CTkFont(size=22, weight="bold")
        self.f_head = ctk.CTkFont(size=14, weight="bold")
        self.f_small = ctk.CTkFont(size=11)
        self.f_mono = ctk.CTkFont(family="Consolas" if sys.platform.startswith("win") else "Menlo", size=11)

        self._build_ui()
        self._refresh_inputs()
        self._update_summary()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_queue)
        # Stay hidden until we have parked the window on the primary monitor.
        # Windows maximizes onto whichever screen the window currently occupies.
        self.withdraw()
        self.after(0, self._open_maximized_on_primary)

    def _open_maximized_on_primary(self):
        x, y = _primary_work_origin() if sys.platform.startswith("win") else (0, 0)
        self.geometry(f"1200x800+{x}+{y}")
        self.deiconify()
        self.update_idletasks()
        self.state("zoomed")

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #
    def _load_settings(self) -> dict:
        s = json.loads(json.dumps(DEFAULT_SETTINGS))
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if k == "models":
                    s["models"].update({mid: bool(on) for mid, on in v.items() if mid in s["models"]})
                else:
                    s[k] = v
        except Exception:
            pass
        return s

    def _save_settings(self):
        data = {
            "prompt": self.prompt_box.get("1.0", "end").strip(),
            "folder": self.folder_var.get().strip(),
            "count": self._read_count(),
            "aspect": self.aspect_var.get(),
            "models": {mid: var.get() for mid, var in self.model_vars.items()},
            "inputs": self.input_paths,
            "geometry": self.geometry(),
        }
        try:
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            print("Could not save settings:", e)

    def _on_close(self):
        self._save_settings()
        self.stop_event.set()
        self.destroy()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _card(self, parent, title: str):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14)
        ctk.CTkLabel(card, text=title, font=self.f_head).pack(anchor="w", padx=16, pady=(14, 4))
        return card

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    # -- sidebar ------------------------------------------------------- #
    def _build_sidebar(self):
        side = ctk.CTkFrame(self, width=360, fg_color="#1E2028", corner_radius=0)
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_propagate(False)

        ctk.CTkLabel(side, text="🎨  Image Studio", font=self.f_title).pack(anchor="w", padx=20, pady=(20, 0))
        ctk.CTkLabel(side, text="Generate with multiple models at once",
                     font=self.f_small, text_color="gray60").pack(anchor="w", padx=22, pady=(0, 12))

        # Models
        models_card = self._card(side, "Models")
        models_card.pack(fill="x", padx=14, pady=(0, 10))
        self.model_vars: dict[str, ctk.BooleanVar] = {}
        for m in MODELS:
            var = ctk.BooleanVar(value=self.settings["models"].get(m["id"], False))
            var.trace_add("write", lambda *_: self._update_summary())
            self.model_vars[m["id"]] = var
            ctk.CTkSwitch(models_card, text=m["name"], variable=var, progress_color=ACCENT,
                          font=ctk.CTkFont(size=13)).pack(anchor="w", padx=16, pady=(8, 0))
            ctk.CTkLabel(models_card, text=f'{m["id"]}  ·  up to {m["max_refs"]} input images',
                         font=self.f_small, text_color="gray55").pack(anchor="w", padx=54)
        ctk.CTkFrame(models_card, height=8, fg_color="transparent").pack()

        # Output
        out_card = self._card(side, "Output")
        out_card.pack(fill="x", padx=14, pady=(0, 10))

        ctk.CTkLabel(out_card, text="Folder", font=self.f_small, text_color="gray60").pack(anchor="w", padx=16)
        row = ctk.CTkFrame(out_card, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(2, 8))
        self.folder_var = ctk.StringVar(value=self.settings["folder"])
        ctk.CTkEntry(row, textvariable=self.folder_var).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="Browse", width=70, fg_color="#343846", hover_color="#42465A",
                      command=self._browse_folder).pack(side="left", padx=(6, 0))

        ctk.CTkLabel(out_card, text="Images per model", font=self.f_small, text_color="gray60").pack(anchor="w", padx=16)
        step = ctk.CTkFrame(out_card, fg_color="transparent")
        step.pack(anchor="w", padx=16, pady=(2, 8))
        self.count_var = ctk.StringVar(value=str(self.settings["count"]))
        self.count_var.trace_add("write", lambda *_: self._update_summary())
        btn_kw = dict(width=32, fg_color="#343846", hover_color="#42465A")
        ctk.CTkButton(step, text="–", command=lambda: self._step_count(-1), **btn_kw).pack(side="left")
        ctk.CTkEntry(step, textvariable=self.count_var, width=56, justify="center").pack(side="left", padx=6)
        ctk.CTkButton(step, text="+", command=lambda: self._step_count(1), **btn_kw).pack(side="left")

        ctk.CTkLabel(out_card, text="Aspect ratio", font=self.f_small, text_color="gray60").pack(anchor="w", padx=16)
        self.aspect_var = ctk.StringVar(
            value=self.settings["aspect"] if self.settings["aspect"] in ASPECT_RATIOS else DEFAULT_ASPECT_LABEL)
        ctk.CTkOptionMenu(out_card, values=ASPECT_RATIOS, variable=self.aspect_var, fg_color="#343846",
                          button_color="#42465A", button_hover_color="#50546B").pack(anchor="w", padx=16, pady=(2, 14))

        # API key
        key_card = self._card(side, "API key")
        key_card.pack(fill="x", padx=14, pady=(0, 10))
        self.key_var = ctk.StringVar(value=os.environ.get("OPENROUTER_API_KEY", ""))
        ctk.CTkEntry(key_card, textvariable=self.key_var, show="•",
                     placeholder_text="sk-or-...").pack(fill="x", padx=16, pady=(0, 4))
        ctk.CTkLabel(key_card, text="Read from OPENROUTER_API_KEY. Never saved to disk.",
                     font=self.f_small, text_color="gray55").pack(anchor="w", padx=16, pady=(0, 12))

    # -- main area ----------------------------------------------------- #
    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=14, pady=14)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(4, weight=1)

        # Prompt
        prompt_card = self._card(main, "Prompt")
        prompt_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.prompt_box = ctk.CTkTextbox(prompt_card, height=90, wrap="word", fg_color="#1B1D24")
        self.prompt_box.pack(fill="x", padx=16, pady=(0, 14))
        self.prompt_box.insert("1.0", self.settings["prompt"])

        # Inputs
        in_card = self._card(main, "Input images")
        in_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        head = ctk.CTkFrame(in_card, fg_color="transparent")
        head.place(relx=1.0, x=-14, y=10, anchor="ne")
        ctk.CTkButton(head, text="＋ Add images", width=110, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=self._add_inputs).pack(side="left")
        ctk.CTkButton(head, text="Clear", width=60, fg_color="#343846", hover_color="#42465A",
                      command=self._clear_inputs).pack(side="left", padx=(6, 0))
        self.inputs_frame = ctk.CTkScrollableFrame(in_card, orientation="horizontal", height=150,
                                                   fg_color="#1B1D24", corner_radius=10)
        self.inputs_frame.pack(fill="x", padx=16, pady=(0, 14))

        # Run controls
        run = ctk.CTkFrame(main, fg_color=CARD, corner_radius=14)
        run.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        run.grid_columnconfigure(2, weight=1)
        self.gen_btn = ctk.CTkButton(run, text="✨  Generate", height=40, width=150, font=self.f_head,
                                     fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self._on_generate)
        self.gen_btn.grid(row=0, column=0, padx=(16, 8), pady=14)
        self.stop_btn = ctk.CTkButton(run, text="Stop", height=40, width=70, fg_color="#7A2E3A",
                                      hover_color="#94404E", state="disabled", command=self._on_stop)
        self.stop_btn.grid(row=0, column=1, padx=(0, 12))
        prog = ctk.CTkFrame(run, fg_color="transparent")
        prog.grid(row=0, column=2, sticky="ew", padx=(0, 16))
        prog.grid_columnconfigure(0, weight=1)
        self.summary_lbl = ctk.CTkLabel(prog, text="", anchor="w", font=self.f_small, text_color="gray65")
        self.summary_lbl.grid(row=0, column=0, sticky="ew")
        self.progress = ctk.CTkProgressBar(prog, progress_color=ACCENT)
        self.progress.set(0)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ctk.CTkButton(run, text="Open folder", width=100, fg_color="#343846", hover_color="#42465A",
                      command=self._open_folder).grid(row=0, column=3, padx=(0, 16))

        # Results
        res_card = self._card(main, "Results  (click to open)")
        res_card.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.results_frame = ctk.CTkScrollableFrame(res_card, orientation="horizontal", height=165,
                                                    fg_color="#1B1D24", corner_radius=10)
        self.results_frame.pack(fill="x", padx=16, pady=(0, 14))
        self.results_hint = ctk.CTkLabel(self.results_frame, text="Generated images will appear here",
                                         text_color="gray50")
        self.results_hint.grid(row=0, column=0, padx=10, pady=40)

        # Log
        log_card = self._card(main, "Log")
        log_card.grid(row=4, column=0, sticky="nsew")
        self.log_box = ctk.CTkTextbox(log_card, font=self.f_mono, fg_color="#1B1D24", state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

    # ------------------------------------------------------------------ #
    # UI actions
    # ------------------------------------------------------------------ #
    def _browse_folder(self):
        d = filedialog.askdirectory(initialdir=self.folder_var.get() or ".")
        if d:
            self.folder_var.set(d)

    def _read_count(self) -> int:
        try:
            return max(1, min(50, int(self.count_var.get())))
        except ValueError:
            return 1

    def _step_count(self, delta: int):
        self.count_var.set(str(max(1, min(50, self._read_count() + delta))))

    def _update_summary(self):
        n_models = sum(v.get() for v in self.model_vars.values())
        count = self._read_count()
        if not self.running:
            self.summary_lbl.configure(
                text=f"{n_models} model(s) × {count} image(s) = {n_models * count} request(s)")

    def _open_folder(self):
        folder = Path(self.folder_var.get().strip() or ".")
        folder.mkdir(parents=True, exist_ok=True)
        open_path(str(folder.resolve()))

    def _log(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{stamp}] {text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # -- input images -------------------------------------------------- #
    def _add_inputs(self):
        files = filedialog.askopenfilenames(
            title="Select input images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"), ("All files", "*.*")])
        for f in files:
            if f not in self.input_paths:
                self.input_paths.append(f)
        self._refresh_inputs()

    def _clear_inputs(self):
        self.input_paths.clear()
        self._refresh_inputs()

    def _remove_input(self, path: str):
        if path in self.input_paths:
            self.input_paths.remove(path)
        self._refresh_inputs()

    def _refresh_inputs(self):
        for w in self.inputs_frame.winfo_children():
            w.destroy()
        if not self.input_paths:
            ctk.CTkLabel(self.inputs_frame, text="No input images — running as text-to-image",
                         text_color="gray50").grid(row=0, column=0, padx=10, pady=40)
            return
        for col, path in enumerate(self.input_paths):
            cell = ctk.CTkFrame(self.inputs_frame, fg_color="#262932", corner_radius=10)
            cell.grid(row=0, column=col, padx=5, pady=4)
            key = (path, INPUT_THUMB_BOX)
            if key not in self._thumb_cache:
                pil = load_thumb_pil(path, INPUT_THUMB_BOX * 2)
                self._thumb_cache[key] = pil_to_ctk(pil, INPUT_THUMB_BOX) if pil else None
            img = self._thumb_cache[key]
            thumb = ctk.CTkFrame(cell, fg_color="transparent",
                                 width=INPUT_THUMB_BOX, height=INPUT_THUMB_BOX)
            thumb.pack(padx=8, pady=(10, 2))
            thumb.pack_propagate(False)
            ctk.CTkLabel(thumb, text="" if img else "?", image=img,
                         width=INPUT_THUMB_BOX, height=INPUT_THUMB_BOX).place(relx=0.5, rely=0.5, anchor="center")
            ctk.CTkButton(thumb, text="×", width=22, height=22, corner_radius=11,
                          font=ctk.CTkFont(size=14, weight="bold"),
                          fg_color="#7A2E3A", hover_color="#94404E",
                          command=lambda p=path: self._remove_input(p)
                          ).place(relx=1.0, rely=0.0, x=6, y=-6, anchor="ne")
            name = Path(path).name
            ctk.CTkLabel(cell, text=name if len(name) <= 14 else name[:11] + "…",
                         font=self.f_small, text_color="gray65").pack(pady=(0, 8))

    # -- results gallery ---------------------------------------------- #
    def _clear_results(self):
        for w in self.results_frame.winfo_children():
            w.destroy()
        self._result_refs.clear()
        self._result_col = 0

    def _add_result(self, path: str, pil_img):
        if self._result_col == 0:
            for w in self.results_frame.winfo_children():
                w.destroy()
        cell = ctk.CTkFrame(self.results_frame, fg_color="#262932", corner_radius=10)
        cell.grid(row=0, column=self._result_col, padx=5, pady=4)
        self._result_col += 1
        img = pil_to_ctk(pil_img, RESULT_THUMB_BOX) if pil_img else None
        self._result_refs.append(img)
        lbl = ctk.CTkLabel(cell, text="" if img else "(preview unavailable)", image=img, cursor="hand2",
                           width=RESULT_THUMB_BOX, height=RESULT_THUMB_BOX)
        lbl.pack(padx=6, pady=(6, 2))
        name = Path(path).name
        cap = ctk.CTkLabel(cell, text=name if len(name) <= 22 else name[:10] + "…" + name[-11:],
                           font=self.f_small, text_color="gray65", cursor="hand2")
        cap.pack(pady=(0, 6))
        for w in (cell, lbl, cap):
            w.bind("<Button-1>", lambda _e, p=path: open_path(p))

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _on_stop(self):
        self.stop_event.set()
        self.stop_btn.configure(state="disabled")
        self._log("Stop requested — pending requests will be skipped (in-flight ones will finish).")

    def _on_generate(self):
        if self.running:
            return
        api_key = self.key_var.get().strip()
        prompt = self.prompt_box.get("1.0", "end").strip()
        selected = [m for m in MODELS if self.model_vars[m["id"]].get()]
        folder_txt = self.folder_var.get().strip()

        if not api_key:
            return messagebox.showwarning("Missing API key", "Set OPENROUTER_API_KEY or enter a key in the sidebar.")
        if not prompt:
            return messagebox.showwarning("Missing prompt", "Please enter a prompt.")
        if not selected:
            return messagebox.showwarning("No models", "Enable at least one model.")
        if not folder_txt:
            return messagebox.showwarning("No folder", "Please specify an output folder.")

        folder = Path(folder_txt)
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return messagebox.showerror("Folder error", str(e))

        count = self._read_count()
        self.count_var.set(str(count))
        aspect = self.aspect_var.get()
        aspect = None if aspect == DEFAULT_ASPECT_LABEL else aspect

        # Pre-assign file numbers so parallel workers never collide.
        tasks = []
        for m in selected:
            slug = model_slug(m["id"])
            start = next_image_number(folder, slug)
            for k in range(start, start + count):
                tasks.append({"model": m, "slug": slug, "number": k})

        self._save_settings()
        self._clear_results()
        self.stop_event.clear()
        self.running = True
        self.total_tasks = len(tasks)
        self.done_tasks = 0
        self.ok_count = 0
        self.total_cost = 0.0
        self.progress.set(0)
        self.summary_lbl.configure(text=f"Running 0/{self.total_tasks}…")
        self.gen_btn.configure(state="disabled", text="Generating…")
        self.stop_btn.configure(state="normal")
        self._log(f"Starting {len(tasks)} request(s) across {len(selected)} model(s) → {folder}")

        threading.Thread(
            target=self._worker,
            args=(api_key, prompt, aspect, folder, tasks, list(self.input_paths)),
            daemon=True,
        ).start()

    # -- background work ---------------------------------------------- #
    def _worker(self, api_key, prompt, aspect, folder, tasks, input_paths):
        q = self.msg_queue
        try:
            data_urls = []
            for p in input_paths:
                try:
                    data_urls.append(file_to_data_url(p))
                except Exception as e:
                    q.put(("log", f"⚠ Could not read input {p}: {e}"))
            if data_urls:
                q.put(("log", f"Encoded {len(data_urls)} input image(s)."))

            warned = set()
            for t in tasks:
                mid, cap = t["model"]["id"], t["model"]["max_refs"]
                if len(data_urls) > cap and mid not in warned:
                    warned.add(mid)
                    q.put(("log", f"⚠ {mid} accepts max {cap} input images; using the first {cap}."))

            with ThreadPoolExecutor(max_workers=max(1, min(len(tasks), MAX_PARALLEL))) as ex:
                futures = [ex.submit(self._generate_one, api_key, prompt, aspect, folder, t,
                                     data_urls[: t["model"]["max_refs"]]) for t in tasks]
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:  # shouldn't happen; _generate_one catches
                        q.put(("progress", False, 0.0))
                        q.put(("log", f"✗ Unexpected error: {e}"))
        finally:
            q.put(("finished",))

    def _generate_one(self, api_key, prompt, aspect, folder, task, refs):
        q = self.msg_queue
        model_id = task["model"]["id"]
        label = f'{task["slug"]}_{task["number"]}'

        if self.stop_event.is_set():
            q.put(("log", f"– {label}: skipped"))
            q.put(("progress", False, 0.0))
            return

        payload = {"model": model_id, "prompt": prompt, "n": 1}
        if aspect:
            payload["aspect_ratio"] = aspect
        if refs:
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": u}} for u in refs]

        q.put(("log", f"→ {label}: request sent…"))
        try:
            resp = requests.post(
                IMAGES_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload, timeout=900)
            try:
                body = resp.json()
            except ValueError:
                body = None

            if resp.status_code != 200 or not isinstance(body, dict) or not body.get("data"):
                msg = None
                if isinstance(body, dict) and isinstance(body.get("error"), dict):
                    msg = body["error"].get("message")
                msg = msg or (resp.text[:300] if resp.text else "empty response")
                q.put(("log", f"✗ {label}: HTTP {resp.status_code} — {msg}"))
                q.put(("progress", False, 0.0))
                return

            item = body["data"][0]
            media_type = item.get("media_type") or "image/png"
            if item.get("b64_json"):
                raw = base64.b64decode(item["b64_json"])
            elif item.get("url"):
                dl = requests.get(item["url"], timeout=300)
                dl.raise_for_status()
                raw = dl.content
                media_type = dl.headers.get("Content-Type", media_type).split(";")[0]
            else:
                q.put(("log", f"✗ {label}: response contained no image data"))
                q.put(("progress", False, 0.0))
                return

            out = folder / f'{label}.{ext_from_media_type(media_type)}'
            out.write_bytes(raw)
            cost = float((body.get("usage") or {}).get("cost") or 0.0)

            q.put(("log", f"✓ {out.name} saved (${cost:.4f})"))
            q.put(("result", str(out), load_thumb_pil(str(out), RESULT_THUMB_BOX * 2)))
            q.put(("progress", True, cost))
        except Exception as e:
            q.put(("log", f"✗ {label}: {e}"))
            q.put(("progress", False, 0.0))

    # -- queue polling (main thread) ---------------------------------- #
    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log(msg[1])
                elif kind == "result":
                    self._add_result(msg[1], msg[2])
                elif kind == "progress":
                    _, ok, cost = msg
                    self.done_tasks += 1
                    self.ok_count += int(ok)
                    self.total_cost += cost
                    self.progress.set(self.done_tasks / max(1, self.total_tasks))
                    self.summary_lbl.configure(
                        text=f"Running {self.done_tasks}/{self.total_tasks}  ·  "
                             f"{self.ok_count} saved  ·  ${self.total_cost:.4f}")
                elif kind == "finished":
                    self.running = False
                    self.gen_btn.configure(state="normal", text="✨  Generate")
                    self.stop_btn.configure(state="disabled")
                    failed = self.done_tasks - self.ok_count
                    self.summary_lbl.configure(
                        text=f"Done — {self.ok_count} saved, {failed} failed/skipped  ·  "
                             f"total cost ${self.total_cost:.4f}")
                    self._log(f"All tasks completed. {self.ok_count}/{self.total_tasks} succeeded, "
                              f"cost ${self.total_cost:.4f}")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


if __name__ == "__main__":
    ImageStudio().mainloop()