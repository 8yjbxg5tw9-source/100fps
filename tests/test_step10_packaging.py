"""Step 10 — standalone packaging, weights manifest, installer config.

No network, no torch, no PyInstaller build here: this suite validates the
packaging *machinery* (resource resolution, hash manifest, spec/installer
sources, fetch helpers). The real frozen build is exercised by
``packaging/build.py`` (smoke-tested) and documented in the README.
"""

from __future__ import annotations

import json
import logging
import os
import py_compile
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_local(name: str):
    """Import packaging/<name>.py by path (setuptools owns `packaging`)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"local_packaging_{name}", REPO_ROOT / "packaging" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ff_mod = _load_local("fetch_ffmpeg")
fw_mod = _load_local("fetch_weights")
build_mod = _load_local("build")
gui_mod = _load_local("gui_launcher")
from pipeline import __version__  # noqa: E402
from pipeline import resources  # noqa: E402
from pipeline.weights_manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    build_manifest,
    file_sha256,
    load_manifest,
    verify_against_manifest,
    write_manifest,
)


@pytest.fixture
def silent_logger() -> logging.Logger:
    logger = logging.getLogger("test-step10")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    return logger


@pytest.fixture
def fresh_frozen_flag():
    """prepare_frozen_environment() is one-shot — reset it around tests."""
    old = resources._PREPARED
    resources._PREPARED = False
    yield
    resources._PREPARED = old


# ---------------------------------------------------------------------------
# pipeline/resources.py — dev mode
# ---------------------------------------------------------------------------
def test_resources_dev_defaults(monkeypatch):
    monkeypatch.delenv("100FPS_WEIGHTS", raising=False)
    monkeypatch.delenv("100FPS_FFMPEG", raising=False)
    assert resources.is_frozen() is False
    assert resources.default_weights_root() == Path("weights")
    assert resources.app_base_dir() == REPO_ROOT
    assert resources.exe_dir() == REPO_ROOT
    assert resources.resource_path("bin", "ffmpeg") == REPO_ROOT / "bin" / "ffmpeg"
    assert resources.prepare_frozen_environment() is False


def test_resources_env_overrides(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("100FPS_WEIGHTS", str(tmp_path / "m"))
    assert resources.default_weights_root() == tmp_path / "m"
    fake = tmp_path / "my-ffmpeg"
    fake.write_bytes(b"x")
    monkeypatch.setenv("100FPS_FFMPEG", str(fake))
    assert resources.find_ffmpeg() == str(fake)
    monkeypatch.setenv("100FPS_FFMPEG", str(tmp_path / "missing"))
    assert resources.find_ffmpeg() != str(tmp_path / "missing")


def test_step_weights_root_defaults_to_dev_dir(silent_logger):
    from pipeline.esrgan.backends import TorchESRGANBackend
    from pipeline.rife.backends import TorchRifeBackend
    from pipeline.step03_interpolate import Step03Interpolate
    from pipeline.step04_upscale import Step04Upscale

    assert Step03Interpolate(logger=silent_logger).weights_root == "weights"
    assert Step04Upscale(logger=silent_logger).weights_root == "weights"
    assert TorchRifeBackend(logger=silent_logger).weights_root is None
    assert TorchESRGANBackend(logger=silent_logger).weights_root is None


# ---------------------------------------------------------------------------
# pipeline/resources.py — frozen simulation
# ---------------------------------------------------------------------------
@pytest.fixture
def frozen_sys(monkeypatch, tmp_path: Path):
    """Pretend we are a PyInstaller bundle rooted at tmp_path."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "100fps"), raising=False)
    return tmp_path


def test_resources_frozen_paths(frozen_sys: Path, fresh_frozen_flag, monkeypatch):
    monkeypatch.delenv("100FPS_WEIGHTS", raising=False)
    assert resources.is_frozen() is True
    assert resources.app_base_dir() == frozen_sys
    assert resources.exe_dir() == frozen_sys
    # Writable models live NEXT TO the exe, never in _MEIPASS.
    assert resources.default_weights_root() == frozen_sys / "models"


def test_prepare_frozen_environment_puts_bin_on_path(
    frozen_sys: Path, fresh_frozen_flag, monkeypatch
):
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    (frozen_sys / "bin").mkdir()
    (frozen_sys / "bin" / name).write_bytes(b"x")
    monkeypatch.setenv("PATH", "/usr/bin")
    assert resources.prepare_frozen_environment() is True
    assert resources.bundled_ffmpeg() == frozen_sys / "bin" / name
    assert resources.find_ffmpeg() == str(frozen_sys / "bin" / name)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(frozen_sys / "bin")
    # Idempotent: second call is a no-op (no PATH duplication).
    assert resources.prepare_frozen_environment() is False
    assert os.environ["PATH"].split(os.pathsep).count(str(frozen_sys / "bin")) == 1


# ---------------------------------------------------------------------------
# weights_manifest.py
# ---------------------------------------------------------------------------
def test_file_sha256_known_vector(tmp_path: Path):
    target = tmp_path / "v.bin"
    target.write_bytes(b"abc")
    assert file_sha256(target) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_manifest_roundtrip_and_tamper(tmp_path: Path):
    root = tmp_path / "models"
    (root / "rife" / "rife4").mkdir(parents=True)
    weights = root / "rife" / "rife4" / "flownet.pkl"
    weights.write_bytes(b"model-bytes")
    manifest = write_manifest(root)
    assert manifest == root / MANIFEST_FILENAME
    data = json.loads(manifest.read_text())
    assert data["app_version"] == __version__
    assert verify_against_manifest(weights, root) is True
    weights.write_bytes(b"model-bytes-TAMPERED")
    assert verify_against_manifest(weights, root) is False


def test_manifest_absent_or_broken_is_legacy_ok(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    weights = root / "x4plus.pth"
    weights.write_bytes(b"model-bytes")
    assert load_manifest(root) == {}
    assert verify_against_manifest(weights, root) is None  # nothing to enforce
    (root / MANIFEST_FILENAME).write_text("not json {{{", encoding="utf-8")
    assert load_manifest(root) == {}
    assert verify_against_manifest(weights, root) is None
    assert verify_against_manifest(tmp_path / "elsewhere.bin", root) is None


def test_build_manifest_skips_itself(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    (root / "a.pth").write_bytes(b"a")
    write_manifest(root)
    entries = build_manifest(root)
    assert set(entries) == {"a.pth"}
    assert entries["a.pth"]["bytes"] == 1


# ---------------------------------------------------------------------------
# ensure_* wiring: manifest match uses cache, mismatch re-downloads
# ---------------------------------------------------------------------------
def test_esrgan_manifest_match_skips_download(tmp_path: Path, silent_logger, monkeypatch):
    from pipeline.esrgan import weights as esrgan_weights

    cached = tmp_path / "esrgan" / "RealESRGAN_x4plus.pth"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"PK" + b"\0" * 1_000_000)
    write_manifest(tmp_path)

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("download must not run on manifest match")

    monkeypatch.setattr(esrgan_weights, "_download", _boom)
    assert esrgan_weights.ensure_esrgan_weights(
        model="x4plus", weights_root=tmp_path, logger=silent_logger
    ) == cached


def test_esrgan_manifest_mismatch_redownloads(tmp_path: Path, silent_logger, monkeypatch):
    from pipeline.esrgan import weights as esrgan_weights
    from pipeline.exceptions import EsrganWeightsError

    cached = tmp_path / "esrgan" / "RealESRGAN_x4plus.pth"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"PK" + b"\0" * 1_000_000)
    write_manifest(tmp_path)
    cached.write_bytes(b"PK-TAMPERED" + b"\0" * 1_000_000)  # still big + PK magic
    calls = []

    def _fake_download(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise RuntimeError("network is down in tests")

    monkeypatch.setattr(esrgan_weights, "_download", _fake_download)
    with pytest.raises(EsrganWeightsError, match="Failed to download"):
        esrgan_weights.ensure_esrgan_weights(
            model="x4plus", weights_root=tmp_path, logger=silent_logger
        )
    assert calls  # tampered cache triggered a re-fetch attempt
    assert not cached.is_file()  # failed re-fetch cleans up


def test_rife_manifest_mismatch_redownloads(tmp_path: Path, silent_logger, monkeypatch):
    from pipeline.exceptions import RifeWeightsError
    from pipeline.rife import weights as rife_weights

    cached = tmp_path / "rife" / "rife4" / "flownet.pkl"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"weights-bytes")
    write_manifest(tmp_path)
    cached.write_bytes(b"weights-bytes-TAMPERED")
    calls = []

    def _fake_download(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise RuntimeError("network is down in tests")

    monkeypatch.setattr(rife_weights, "_download_drive_file", _fake_download)
    with pytest.raises(RifeWeightsError, match="Failed to download"):
        rife_weights.ensure_weights(
            version="4", weights_root=tmp_path, logger=silent_logger
        )
    assert calls


# ---------------------------------------------------------------------------
# Frozen launcher behaviour in main.py / app.py / gui_launcher.py
# ---------------------------------------------------------------------------
def test_frozen_no_args_opens_gui(monkeypatch, fresh_frozen_flag):
    import app
    import main as cli

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "argv", ["100fps"], raising=False)
    seen: Dict[str, Any] = {}

    def _fake_launch(**kwargs: Any) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(app, "launch_ui", _fake_launch)
    assert cli.main(None) == 0
    assert seen.get("inbrowser") is True


def test_cli_no_args_dev_returns_2(capsys):
    import main as cli

    assert cli.main([]) == 2  # normal "missing --input" path, not the GUI hook


def test_cli_version_flag(capsys):
    import main as cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_gui_launcher_calls_launch_ui(monkeypatch):
    import app

    seen: Dict[str, Any] = {}
    monkeypatch.setattr(app, "launch_ui", lambda **k: seen.update(k))
    assert gui_mod.main() == 0
    assert seen.get("inbrowser") is True


def test_app_launch_ui_accepts_inbrowser():
    import inspect

    import app

    assert "inbrowser" in inspect.signature(app.launch_ui).parameters


# ---------------------------------------------------------------------------
# PyInstaller spec file
# ---------------------------------------------------------------------------
def test_spec_compiles_and_references_real_entries():
    spec = REPO_ROOT / "packaging" / "100fps.spec"
    assert spec.is_file()
    py_compile.compile(str(spec), doraise=True)
    text = spec.read_text(encoding="utf-8")
    assert "ROOT / \"main.py\"" in text
    assert "gui_launcher.py" in text
    assert "collect_all" in text  # defensive optional-dep collection
    assert "FPS_TARGET" in text
    assert "console=False" in text  # windowed GUI exe
    assert "upx" in text.lower()
    assert (REPO_ROOT / "main.py").is_file()
    assert (REPO_ROOT / "packaging" / "gui_launcher.py").is_file()
    assert (REPO_ROOT / "packaging" / "assets" / "icon.ico").is_file()


# ---------------------------------------------------------------------------
# Inno Setup script
# ---------------------------------------------------------------------------
def test_iss_version_matches_package():
    iss = REPO_ROOT / "packaging" / "innosetup" / "100fps.iss"
    assert iss.is_file()
    text = iss.read_text(encoding="utf-8")
    match = re.search(r'#define\s+MyAppVersion\s+"([^"]+)"', text)
    assert match, "MyAppVersion define missing from the .iss"
    assert match.group(1) == __version__


def test_iss_has_shortcuts_cuda_check_and_uninstaller():
    text = (REPO_ROOT / "packaging" / "innosetup" / "100fps.iss").read_text(
        encoding="utf-8"
    )
    assert "PrivilegesRequired=lowest" in text  # per-user, writable install
    assert "desktopicon" in text  # desktop shortcut task
    assert "{group}" in text  # start-menu shortcuts
    assert "nvidia-smi" in text  # CUDA/driver preflight warning
    assert "{uninstallexe}" in text  # uninstaller entry
    assert "ArchitecturesAllowed=x64compatible" in text


# ---------------------------------------------------------------------------
# Icon assets
# ---------------------------------------------------------------------------
def test_icon_assets_exist_and_load():
    pytest.importorskip("PIL")
    from PIL import Image

    assets = REPO_ROOT / "packaging" / "assets"
    assert (assets / "icon.png").is_file()
    assert (assets / "icon.ico").is_file()
    with Image.open(assets / "icon.ico") as ico:
        assert (256, 256) in (ico.info.get("sizes") or set())
    with Image.open(assets / "icon.png") as png:
        assert png.size == (1024, 1024)


# ---------------------------------------------------------------------------
# fetch_weights.py (no network — selection logic only)
# ---------------------------------------------------------------------------
def test_parse_selection():
    assert fw_mod.parse_selection("all", ["a", "b"]) == ["a", "b"]
    assert fw_mod.parse_selection("none", ["a", "b"]) == []
    assert fw_mod.parse_selection("b,a", ["a", "b"]) == ["b", "a"]
    with pytest.raises(ValueError, match="Unknown"):
        fw_mod.parse_selection("zzz", ["a", "b"])


def test_fetch_weights_list_and_empty_selection(capsys):
    assert fw_mod.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "RIFE" in out and "Real-ESRGAN" in out
    assert fw_mod.main(["--rife", "none", "--esrgan", "none"]) == 2


# ---------------------------------------------------------------------------
# fetch_ffmpeg.py (no network — archive handling only)
# ---------------------------------------------------------------------------
def test_detect_platform_and_sources():
    assert ff_mod.detect_platform() in ("windows", "linux", "macos")
    assert set(ff_mod.SOURCES) == {"windows", "linux", "macos"}
    for source in ff_mod.SOURCES.values():
        assert source["url"].startswith("https://")
        assert source["members"]


def test_extract_members_zip_and_tar(tmp_path: Path):
    out = tmp_path / "bin"
    out.mkdir()
    archive = tmp_path / "ffmpeg.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("ffmpeg-9-essentials_build/bin/ffmpeg.exe", b"fake-exe")
    written = ff_mod._extract_members(
        archive, "zip", {"*/bin/ffmpeg.exe": "ffmpeg.exe"}, out
    )
    assert written == [out / "ffmpeg.exe"]
    assert (out / "ffmpeg.exe").read_bytes() == b"fake-exe"

    tarball = tmp_path / "ffmpeg.tar.xz"
    with tarfile.open(tarball, "w:xz") as tf:
        import io

        payload = b"fake-bin"
        info = tarfile.TarInfo("ffmpeg-9-amd64-static/ffmpeg")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    written = ff_mod._extract_members(
        tarball, "tarxz", {"ffmpeg-*-amd64-static/ffmpeg": "ffmpeg"}, out
    )
    assert written == [out / "ffmpeg"]
    with pytest.raises(FileNotFoundError, match="no member matching"):
        ff_mod._extract_members(archive, "zip", {"*/nope": "x"}, out)


def test_from_local_and_verify_rejects_toys(tmp_path: Path):
    src = tmp_path / "sys"
    src.mkdir()
    (src / "ffmpeg").write_bytes(b"tiny")
    (src / "ffmpeg-linux-x86_64-v7.0.2").write_bytes(b"tiny-too")
    out = tmp_path / "bin"
    out.mkdir()
    copied = ff_mod._from_local(src, out)
    assert copied == [out / "ffmpeg"]  # exact name wins, no duplicates
    out2 = tmp_path / "bin2"
    out2.mkdir()
    src2 = tmp_path / "imageio-style"
    src2.mkdir()
    (src2 / "ffmpeg-linux-x86_64-v7.0.2").write_bytes(b"tiny")
    assert ff_mod._from_local(src2, out2) == [out2 / "ffmpeg"]  # renamed
    with pytest.raises(ValueError, match="suspiciously small"):
        ff_mod.verify_binary(out / "ffmpeg")
    with pytest.raises(FileNotFoundError, match="No ffmpeg"):
        ff_mod._from_local(tmp_path / "empty-dir", out)


# ---------------------------------------------------------------------------
# build.py (no real build — plumbing only)
# ---------------------------------------------------------------------------
def test_build_version_and_platform():
    assert build_mod.read_version() == __version__
    assert build_mod.platform_tag() in ("win64", "linux", "macos")
    args = build_mod.build_parser().parse_args([])
    assert args.target == "onedir"
    assert args.ffmpeg == "auto"
    assert args.weights == "none"


def test_check_prereqs_reports_missing_tool(monkeypatch):
    monkeypatch.setitem(sys.modules, "PyInstaller", None)  # simulate absence
    with pytest.raises(RuntimeError, match="pip install PyInstaller"):
        build_mod.check_prereqs(auto_install=False)


def test_build_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        build_mod.main(["--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# README ships the distribution docs (Step 10 §4)
# ---------------------------------------------------------------------------
def test_readme_has_distribution_docs():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for heading in (
        "## Step 10",
        "Aparat tələbləri",
        "Problem həlli",
        "Quraşdırma və İstifadə",
    ):
        assert heading in readme, f"README missing section: {heading}"
