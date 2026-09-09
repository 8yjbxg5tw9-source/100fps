# -*- mode: python; coding: utf-8 -*-
"""PyInstaller build for 100fps (Step 10).

Builds TWO executables sharing one bundle: ``100fps`` (console CLI,
``main.py``) and ``100fps-gui`` (windowed, ``gui_launcher.py``).

Designed to be defensive: heavy optional dependencies (torch, gradio,
onnxruntime) are collected when PRESENT and skipped with a warning when
absent, so a smoke build works on a bare machine while a full build on a
GPU box picks up CUDA DLLs automatically (via pyinstaller-hooks-contrib).

Do not invoke directly — use ``python packaging/build.py``, which prepares
``packaging/vendor/`` (ffmpeg binaries, optional weights) first.

Env knobs read here:
    FPS_TARGET=onefile   single-file EXE instead of the default one-dir.
"""

import os
import sys
from pathlib import Path

SPEC_DIR = Path(SPECPATH)  # noqa: F821 - provided by PyInstaller
ROOT = SPEC_DIR.parent
VENDOR = SPEC_DIR / "vendor"
ASSETS = SPEC_DIR / "assets"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ONEFILE = os.environ.get("FPS_TARGET", "onedir").lower() == "onefile"
ON_WINDOWS = sys.platform == "win32"

# -- Defensive collectors -------------------------------------------------------
# Missing optional packages warn (never fail): torch/gradio simply won't be
# in the bundle, and the app's own error messages guide the user.


def _try_collect_all(pkg):
    import importlib.util

    # NOTE: collect_all() does NOT raise for missing packages (it warns and
    # returns empties), so check importability explicitly for honest logging.
    if importlib.util.find_spec(pkg) is None:
        print(f"100fps.spec: WARNING — optional package {pkg!r} not installed; "
              f"skipped (install it before building for a full release).")
        return [], [], []
    try:
        from PyInstaller.utils.hooks import collect_all

        datas, binaries, hidden = collect_all(pkg)
        print(f"100fps.spec: bundled optional package {pkg!r}.")
        return datas, binaries, hidden
    except Exception as exc:  # noqa: BLE001 - optional dep, warn and continue
        print(f"100fps.spec: WARNING — optional package {pkg!r} not bundled: {exc}")
        return [], [], []


extra_datas = []
extra_binaries = []
extra_hidden = []
for _pkg in ("torch", "torchvision", "gradio", "onnxruntime", "cv2"):
    _d, _b, _h = _try_collect_all(_pkg)
    extra_datas += _d
    extra_binaries += _b
    extra_hidden += _h

# -- Vendored ffmpeg/ffprobe -> bin/ inside the bundle ---------------------------
# (pipeline/resources.py puts this dir on PATH at startup, so no code changes
# are needed to use the shipped binaries.)
if ON_WINDOWS:
    _ffmpeg_names = ("ffmpeg.exe", "ffprobe.exe")
else:
    _ffmpeg_names = ("ffmpeg", "ffprobe")
_plat = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
_vendor_bin = VENDOR / "bin" / _plat
for _name in _ffmpeg_names:
    _candidate = _vendor_bin / _name
    if _candidate.is_file():
        extra_binaries.append((str(_candidate), "bin"))
        print(f"100fps.spec: bundled {_candidate.relative_to(ROOT)}.")
    else:
        print(f"100fps.spec: WARNING — {_candidate.relative_to(ROOT)} missing; "
              f"run packaging/fetch_ffmpeg.py (app falls back to system "
              f"binaries / OpenCV probing).")

# -- Static files ---------------------------------------------------------------
for _doc in ("dist-readme.txt", "THIRD_PARTY_LICENSES.txt"):
    _candidate = SPEC_DIR / _doc
    if _candidate.is_file():
        extra_datas.append((str(_candidate), "."))
    else:  # pragma: no cover - build.py guarantees these
        print(f"100fps.spec: WARNING — {SPEC_DIR.name}/{_doc} missing.")

hiddenimports = sorted(
    set(
        extra_hidden
        + [
            # Dynamically imported inside functions (never found statically).
            "tqdm",
            "cv2",
            "PIL",
            "numpy",
            "onnxruntime",
            "gradio",
            "onnx",
            # Our own lazy imports (gui_launcher needs no entry here: it is
            # analysed directly as the second EXE's script).
            "pipeline.rife.vendor.model",
            "pipeline.esrgan.vendor.upstream.rrdbnet",
            "app",
        ]
    )
)

excludes = [
    "FixTk",
    "tcl",
    "tk",
    "_tkinter",
    "unittest",
    "pydoc",
    "doctest",
]

# UPX corrupts torch/CUDA DLLs — keep it off even though it would shrink us.
UPX_OFF = False

# Both entries share one bundle: each Analysis gets the SAME inputs (no MERGE —
# MERGE would give the GUI exe onefile-extraction semantics). Pure modules
# are duplicated across the two small .pyz archives; binaries/datas land on
# disk once via the single COLLECT below.
cli_analysis = Analysis(  # noqa: F821 - PyInstaller spec globals
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=extra_binaries,
    datas=extra_datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
gui_analysis = Analysis(  # noqa: F821
    [str(SPEC_DIR / "gui_launcher.py")],
    pathex=[str(ROOT)],
    binaries=extra_binaries,
    datas=extra_datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)

cli_pyz = PYZ(cli_analysis.pure)  # noqa: F821
gui_pyz = PYZ(gui_analysis.pure)  # noqa: F821

_icon = str(ASSETS / "icon.ico") if ON_WINDOWS else None

# Onefile mode ships the CLI only: a second onefile GUI exe would duplicate
# the whole torch/CUDA payload (~GBs). The GUI ships with onedir/installer.
cli_exe = EXE(  # noqa: F821
    cli_pyz,
    cli_analysis.scripts,
    ([] if ONEFILE else cli_analysis.binaries + cli_analysis.datas),
    exclude_binaries=not ONEFILE,
    name="100fps",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=UPX_OFF,
    console=True,
    icon=_icon,
)

if ONEFILE:
    print("100fps.spec: onefile target — CLI only (use onedir for the GUI).")
else:
    gui_exe = EXE(  # noqa: F821
        gui_pyz,
        gui_analysis.scripts,
        [],
        exclude_binaries=True,
        name="100fps-gui",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=UPX_OFF,
        console=False,  # no console window on double-click (Windows/macOS)
        icon=_icon,
        disable_windowed_traceback=False,
    )
    coll = COLLECT(  # noqa: F821
        cli_exe,
        cli_analysis.binaries,
        cli_analysis.datas,
        gui_exe,
        strip=False,
        upx=UPX_OFF,
        name="100fps",
    )
    if sys.platform == "darwin":  # pragma: no cover - macOS-only stanza
        app = BUNDLE(  # noqa: F821
            coll,
            name="100fps.app",
            icon=str(ASSETS / "icon.ico"),
            bundle_identifier="com.100fps.pipeline",
            info_plist={"NSHighResolutionCapable": "True"},
        )
