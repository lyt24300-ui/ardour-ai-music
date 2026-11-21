"""Standalone desktop window launcher for Ardour AI Music Studio.

Triggered by the single 'AI Music' button inside Ardour's AI Music tab
(or run directly via `python -m ai_music_studio.launcher`).
Ensures single-instance behavior: if already running, brings the window
to front instead of spawning duplicate windows.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from .config import DATA_DIR, STUDIO_HOST, STUDIO_PORT


def _ensure_stdio() -> None:
    """Attach stdout/stderr to a UTF-8 log file when launched via pythonw.exe (no console)."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DATA_DIR / "studio.log"
    stream = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _is_server_alive(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=1.2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _trigger_focus(url: str) -> bool:
    try:
        req = urllib.request.Request(f"{url}/api/focus", method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _find_app_browser() -> str | None:
    """Locate Edge or Chrome for native standalone --app desktop window mode."""
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    for name in ("msedge", "microsoft-edge", "google-chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _open_standalone_window(url: str) -> subprocess.Popen | None:
    browser = _find_app_browser()
    profile_dir = DATA_DIR / "window_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    if browser:
        cmd = [
            browser,
            f"--app={url}",
            f"--user-data-dir={profile_dir}",
            "--window-size=1420,880",
            "--disable-features=TranslateUI",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        return subprocess.Popen(cmd)
    webbrowser.open(url)
    return None


def main() -> None:
    _ensure_stdio()
    parser = argparse.ArgumentParser(description="Ardour AI Music Studio Launcher")
    parser.add_argument("--host", default=STUDIO_HOST)
    parser.add_argument("--port", type=int, default=STUDIO_PORT)
    parser.add_argument("--mcp-url", default="")
    parser.add_argument("--no-window", action="store_true", help="Run backend server only")
    args = parser.parse_args()

    if args.mcp_url:
        os.environ["ARDOUR_MCP_URL"] = args.mcp_url

    url = _base_url(args.host, args.port)

    # Single-instance check: focus existing window if server is already alive
    if _is_server_alive(url):
        if not args.no_window:
            if not _trigger_focus(url):
                _open_standalone_window(url)
        return

    import uvicorn
    from .server import app

    server_cfg = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning", use_colors=False)
    server = uvicorn.Server(server_cfg)

    if args.no_window:
        server.run()
        return

    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait for FastAPI to accept connections
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if _is_server_alive(url):
            break
        time.sleep(0.15)

    proc = _open_standalone_window(url)
    if proc is not None:
        started_at = time.time()
        try:
            proc.wait()
        except KeyboardInterrupt:
            server.should_exit = True
            return
        if time.time() - started_at > 3.0:
            server.should_exit = True
        else:
            try:
                t.join()
            except KeyboardInterrupt:
                server.should_exit = True
    else:
        try:
            t.join()
        except KeyboardInterrupt:
            server.should_exit = True


if __name__ == "__main__":
    main()
