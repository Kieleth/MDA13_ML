#!/usr/bin/env python3
"""
Cañadata — setup check.

Run from the project root:

    python setup_check.py

If every check is green, you're ready for the workshop. If anything is red,
read the "fix this by:" line right below the failure and follow it. Yellow
is a soft warning that doesn't block the workshop but is worth fixing.

Hints are platform-aware — macOS and Windows users get tailored advice.

The OpenAI ping costs about $0.0001 (one round-trip with max_tokens=1). If you
don't want to spend that yet, skip it: `python setup_check.py --skip-api`.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PLATFORM = platform.system()  # 'Darwin', 'Windows', 'Linux'
IS_MAC = PLATFORM == "Darwin"
IS_WIN = PLATFORM == "Windows"
IS_LINUX = PLATFORM == "Linux"

# Disable colors on Windows by default unless we detect a sane terminal.
# Modern Windows Terminal and PowerShell 7+ handle ANSI fine. Old cmd.exe doesn't.
USE_COLOR = sys.stdout.isatty() and (not IS_WIN or os.environ.get("WT_SESSION") or os.environ.get("TERM"))

GREEN = "\033[32m" if USE_COLOR else ""
RED = "\033[31m" if USE_COLOR else ""
YELLOW = "\033[33m" if USE_COLOR else ""
DIM = "\033[2m" if USE_COLOR else ""
BOLD = "\033[1m" if USE_COLOR else ""
RESET = "\033[0m" if USE_COLOR else ""

failures = 0
warnings = 0


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str, hint: str | None = None) -> None:
    global failures
    failures += 1
    print(f"  {RED}✗{RESET} {msg}")
    if hint:
        print(f"    {DIM}fix this by: {hint}{RESET}")


def warn(msg: str, hint: str | None = None) -> None:
    global warnings
    warnings += 1
    print(f"  {YELLOW}⚠{RESET} {msg}")
    if hint:
        print(f"    {DIM}fix this by: {hint}{RESET}")


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


# ── Platform info ───────────────────────────────────────────────────────────

def print_header() -> None:
    print(f"\n{BOLD}Cañadata setup check{RESET}")
    print(f"{DIM}working dir:  {ROOT}{RESET}")
    bits = []
    if IS_MAC:
        bits.append(f"macOS {platform.mac_ver()[0]} ({platform.machine()})")
    elif IS_WIN:
        bits.append(f"Windows {platform.release()} ({platform.machine()})")
    else:
        bits.append(f"{PLATFORM} ({platform.machine()})")
    bits.append(f"Python {sys.version.split()[0]}")
    if os.environ.get("CONDA_DEFAULT_ENV"):
        bits.append(f"conda env: {os.environ['CONDA_DEFAULT_ENV']}")
    elif os.environ.get("VIRTUAL_ENV"):
        bits.append(f"venv: {Path(os.environ['VIRTUAL_ENV']).name}")
    if IS_WIN:
        shell = detect_windows_shell()
        if shell:
            bits.append(f"shell: {shell}")
    print(f"{DIM}platform:     {' · '.join(bits)}{RESET}")


def detect_windows_shell() -> str | None:
    """Best-effort detection of which Windows shell is hosting Python."""
    if not IS_WIN:
        return None
    # PowerShell sets PSModulePath; modern PowerShell sets POSH_*
    if os.environ.get("POSH_THEME") or os.environ.get("POWERSHELL_TELEMETRY_OPTOUT"):
        return "PowerShell 7+"
    if os.environ.get("PSModulePath"):
        return "PowerShell"
    if os.environ.get("CONDA_PROMPT_MODIFIER") and "Anaconda" in (os.environ.get("CONDA_EXE", "") or ""):
        return "Anaconda Prompt"
    if os.environ.get("COMSPEC", "").lower().endswith("cmd.exe"):
        return "cmd.exe"
    return None


# ── Subprocess probe helper ─────────────────────────────────────────────────

def run_version(cmd: list[str], timeout: float = 5.0) -> tuple[bool, str]:
    """Run `cmd --version` and return (success, first_line)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return False, "command not found"
    except subprocess.TimeoutExpired:
        return False, "timed out"
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"
    output = (result.stdout or result.stderr or "").strip().splitlines()
    first_line = output[0] if output else ""
    return result.returncode == 0, first_line


# ── Checks ──────────────────────────────────────────────────────────────────

def check_python() -> None:
    section("1. Python")
    version = sys.version.split()[0]
    if sys.version_info >= (3, 10):
        ok(f"Python {version}")
    else:
        if IS_WIN:
            hint = "uninstall and reinstall Python 3.12 from python.org with the 'Add Python to PATH' checkbox"
        elif IS_MAC:
            hint = "create a fresh env: conda create -n mda13_ml python=3.12 -y && conda activate mda13_ml"
        else:
            hint = "create a fresh env: conda create -n mda13_ml python=3.12 -y && conda activate mda13_ml"
        fail(f"Python {version} (need 3.10+)", hint)

    # Conda / venv hint — surface if neither is active
    if not (os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")):
        if IS_WIN:
            warn(
                "no conda/venv environment active",
                "open Anaconda Prompt, then `conda activate mda13_ml`. PowerShell needs `conda init powershell` once.",
            )
        else:
            warn(
                "no conda/venv environment active",
                "`conda activate mda13_ml` (or activate your venv)",
            )


def check_imports() -> None:
    section("2. Python dependencies")
    required = [
        ("streamlit", "pip install streamlit"),
        ("pandas", "pip install pandas"),
        ("numpy", "pip install numpy"),
        ("sklearn", "pip install scikit-learn"),
        ("statsmodels", "pip install statsmodels"),
        ("prophet", "pip install prophet"),
        ("openai", "pip install openai"),
        ("dotenv", "pip install python-dotenv"),
        ("joblib", "pip install joblib"),
        ("matplotlib", "pip install matplotlib"),
        ("seaborn", "pip install seaborn"),
    ]
    for mod_name, hint in required:
        try:
            importlib.import_module(mod_name)
            ok(f"import {mod_name}")
        except ImportError as e:
            fail(f"import {mod_name} — {e}", hint)

    # xgboost: importing succeeds even without libomp on macOS, but instantiation fails.
    # Probe end-to-end so the failure surfaces here, not mid-notebook.
    try:
        import numpy as np
        from xgboost import XGBClassifier
        XGBClassifier(n_estimators=2, max_depth=2).fit(
            np.array([[0, 0], [1, 1]]), np.array([0, 1])
        )
        ok("xgboost (end-to-end probe)")
    except ImportError:
        warn(
            "xgboost not installed",
            "pip install xgboost. Optional — the homework notebook falls back to RandomForest.",
        )
    except Exception as e:
        msg = str(e).splitlines()[0]
        if "libomp" in str(e).lower() or "OpenMP" in str(e):
            if IS_MAC:
                hint = "brew install libomp (and if no brew: install Homebrew from https://brew.sh)"
            else:
                hint = "ensure OpenMP runtime is installed for your system"
            warn(
                "xgboost installed but cannot run (OpenMP runtime missing)",
                f"{hint}. Optional — the homework notebook falls back to RandomForest.",
            )
        else:
            warn(
                f"xgboost runtime error: {msg}",
                "the homework notebook falls back to RandomForest, you can ignore",
            )


def check_cli_tools() -> None:
    section("3. CLI tools on PATH")
    cli_tools = [
        (["git", "--version"], "git",
         "Windows: install from https://git-scm.com. macOS: `xcode-select --install`."),
        (["streamlit", "--version"], "streamlit",
         "with the env active, `pip install streamlit` should add it to PATH."),
        (["jupyter", "--version"], "jupyter",
         "`pip install jupyter`. If installed but not on PATH, ensure the env is activated."),
    ]
    for cmd, label, hint in cli_tools:
        success, output = run_version(cmd)
        if success:
            ok(f"{label}: {output}")
        elif "command not found" in output or "not recognized" in output.lower():
            fail(f"{label}: not on PATH", hint)
        else:
            warn(f"{label}: {output}", hint)


def check_data_files() -> None:
    section("4. Data files")
    expected = [
        ROOT / "data" / "canadata_leads.csv",
        ROOT / "data" / "canadata_holdout.csv",
    ]
    for path in expected:
        if path.exists():
            ok(f"{path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")
        else:
            fail(
                f"{path.relative_to(ROOT)} missing",
                "el CSV debería venir en el repo. Si no está, vuelve a clonar o pídeselo a Luis por el chat.",
            )

    cleaned = ROOT / "data" / "canadata_leads_clean.csv"
    if cleaned.exists():
        ok(f"{cleaned.relative_to(ROOT)} ({cleaned.stat().st_size // 1024} KB)")
    else:
        warn(
            "data/canadata_leads_clean.csv missing",
            "run pre_class/1_classical_models.ipynb (the warm-up) end to end — it produces this file",
        )


def check_env_file() -> tuple[bool, str | None]:
    section("5. Environment & API key")
    env_path = ROOT / ".env"
    if not env_path.exists():
        fail(
            ".env file missing at project root",
            "create .env with: OPENAI_API_KEY=sk-... (key shared by Luis)",
        )
        return False, None

    ok(".env file exists")

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        warn("python-dotenv not installed, cannot load .env", "pip install python-dotenv")
        return False, None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        fail(
            "OPENAI_API_KEY not set",
            "add the line OPENAI_API_KEY=sk-... to .env (key shared by Luis)",
        )
        return False, None

    if not api_key.startswith("sk-"):
        warn(
            f"OPENAI_API_KEY format looks off (starts with {api_key[:3]}...)",
            "double-check the key Luis shared, no quotes or whitespace around the value",
        )

    ok(f"OPENAI_API_KEY set ({api_key[:7]}...{api_key[-4:]})")
    return True, api_key


def check_api_ping(api_key: str) -> None:
    section("6. OpenAI API ping (live, ~$0.0001)")
    try:
        from openai import OpenAI
    except ImportError:
        fail("openai package not installed", "pip install openai")
        return

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        usage = resp.usage
        ok(
            f"API responded ({resp.model}, "
            f"{usage.prompt_tokens} in / {usage.completion_tokens} out tokens)"
        )
    except Exception as e:
        msg = str(e)
        if "Incorrect API key" in msg or "invalid_api_key" in msg.lower():
            fail(
                "API key rejected",
                "the key is wrong. Re-paste it from Luis without quotes or whitespace",
            )
        elif "model_not_found" in msg.lower() or "does not exist" in msg.lower():
            fail(
                "model gpt-4.1-mini not available on this account",
                "tell Luis — the model id may have moved, we'll pin a working one",
            )
        elif "rate" in msg.lower():
            warn(
                "rate-limited on the ping",
                "wait 30s and re-run; not a real failure",
            )
        elif "timeout" in msg.lower() or "connection" in msg.lower():
            fail(
                "could not reach OpenAI",
                "check internet, VPN, corporate proxy. Then re-run.",
            )
        else:
            fail(f"API call failed: {msg.splitlines()[0]}", "check internet, then re-run")


def check_trained_models() -> None:
    section("7. Trained models (homework output)")
    models_dir = ROOT / "session1" / "models"
    expected = [
        "classifier.pkl",
        "regressor.pkl",
        "clusterer.pkl",
        "timeseries.pkl",
    ]
    if not models_dir.exists():
        warn(
            f"{models_dir.relative_to(ROOT)} does not exist yet",
            "run pre_class/1_classical_models.ipynb (under 3 min) to produce the artifacts",
        )
        return

    if not os.access(models_dir, os.W_OK):
        warn(
            f"{models_dir.relative_to(ROOT)} is not writable",
            "check folder permissions; on Windows run terminal as Administrator only if necessary",
        )

    missing = [name for name in expected if not (models_dir / name).exists()]
    if missing:
        warn(
            f"missing artifacts: {', '.join(missing)}",
            "run pre_class/1_classical_models.ipynb to produce them",
        )
    else:
        for name in expected:
            ok(f"session1/models/{name}")


def check_platform_specific() -> None:
    section("8. Platform-specific")
    if IS_MAC:
        # Brew presence (we hint at it for libomp)
        success, output = run_version(["brew", "--version"])
        if success:
            ok(f"Homebrew: {output}")
        else:
            warn(
                "Homebrew not found",
                "install from https://brew.sh — needed if you want xgboost (libomp). Not blocking.",
            )

    if IS_WIN:
        # Encoding sanity: Windows defaults to cp1252 in old cmd.exe which can break unicode.
        encoding = sys.stdout.encoding.lower() if sys.stdout.encoding else "unknown"
        if "utf" in encoding:
            ok(f"console encoding: {encoding}")
        else:
            warn(
                f"console encoding: {encoding} (not UTF-8)",
                "set `chcp 65001` in cmd.exe, or `$env:PYTHONIOENCODING='utf-8'` in PowerShell, "
                "to avoid Unicode errors when running Streamlit.",
            )

        shell = detect_windows_shell()
        if shell == "PowerShell" and not os.environ.get("CONDA_DEFAULT_ENV"):
            warn(
                "PowerShell detected without an active conda env",
                "in PowerShell, run `conda init powershell` once, then restart the shell. "
                "Or use Anaconda Prompt instead.",
            )

    if IS_LINUX:
        ok(f"Linux ({platform.machine()}) — should generally just work")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip the live OpenAI ping (saves $0.0001 and a network round-trip).",
    )
    args = parser.parse_args()

    print_header()

    check_python()
    check_imports()
    check_cli_tools()
    check_data_files()
    env_ok, api_key = check_env_file()
    if env_ok and api_key and not args.skip_api:
        check_api_ping(api_key)
    elif args.skip_api:
        section("6. OpenAI API ping")
        print(f"  {DIM}(skipped){RESET}")
    check_trained_models()
    check_platform_specific()

    print()
    if failures == 0 and warnings == 0:
        print(f"{GREEN}{BOLD}All green. You're ready.{RESET}\n")
        return 0
    if failures == 0:
        print(f"{YELLOW}{BOLD}Setup OK, with {warnings} warning(s) above.{RESET}")
        print(f"{DIM}Warnings won't block the workshop, but address them when you can.{RESET}\n")
        return 0
    print(f"{RED}{BOLD}{failures} check(s) failed.{RESET} Fix the items above and re-run.")
    print(f"{DIM}Stuck? See TROUBLESHOOTING.md (mac + Windows pain points with copy-pasteable fixes).{RESET}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
