"""Windowed GUI entry point for frozen builds (Step 10).

Built as ``100fps-gui`` (``100fps-gui.exe`` on Windows, no console window).
Double-click behaviour: prepare the bundle environment, start the Gradio
WebUI, open the browser.
"""

from pipeline.resources import prepare_frozen_environment


def main() -> int:
    prepare_frozen_environment()
    import app

    app.launch_ui(inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
