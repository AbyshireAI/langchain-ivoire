"""Optional settings file for the remote client, kept out of the code.

The file is JSON and is never part of the package. It is looked up, in order, at:

1. the path in ``IVOIRE_SETTINGS`` (``off``, ``0``, ``none`` or an empty value turn the lookup off);
2. ``%APPDATA%\\langchain-ivoire\\settings.json`` on Windows, else
   ``$XDG_CONFIG_HOME/langchain-ivoire/settings.json`` (``~/.config/langchain-ivoire/settings.json``).

Shape (every key is optional)::

    {"remote": {"style_aliases": {}}}

Arguments passed in code win over environment variables, which win over the file.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

from langchain_ivoire.exceptions import IvoireConfigError

__all__ = ["SETTINGS_ENV", "SettingsLike", "default_settings_path", "load_settings", "section"]

SETTINGS_ENV = "IVOIRE_SETTINGS"
_OFF = {"", "0", "off", "none", "false", "no"}

SettingsLike = Union[str, "os.PathLike[str]", Mapping[str, Any], None]
"""A settings file path, the settings themselves, or ``None`` (look the file up)."""


def default_settings_path() -> Path:
    """Where the settings file is looked for when ``IVOIRE_SETTINGS`` is not set."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "langchain-ivoire" / "settings.json"


def _settings_path() -> Path | None:
    env = os.environ.get(SETTINGS_ENV)
    if env is not None:
        if env.strip().lower() in _OFF:
            return None
        return Path(env.strip()).expanduser()
    p = default_settings_path()
    return p if p.is_file() else None


def load_settings(source: SettingsLike = None) -> dict[str, Any]:
    """The settings: ``source`` itself if it is a mapping, else the JSON file at ``source`` or at
    the looked-up location (``{}`` when there is none).

    Raises:
        IvoireConfigError: the file named by ``source`` or ``IVOIRE_SETTINGS`` is missing, is not
            valid JSON, or does not hold a JSON object.
    """
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source).expanduser() if source is not None else _settings_path()
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise IvoireConfigError(f"settings file not found: {path}") from None
    except (OSError, ValueError) as e:
        raise IvoireConfigError(
            f"could not read the settings file {path} ({type(e).__name__})"
        ) from None
    if not isinstance(data, dict):
        raise IvoireConfigError(f"the settings file {path} must hold a JSON object")
    return data


def section(settings: Mapping[str, Any], name: str) -> dict[str, Any]:
    """One section (``"local"`` or ``"remote"``) of the settings, ``{}`` if absent."""
    s = settings.get(name)
    if s is None:
        return {}
    if not isinstance(s, Mapping):
        raise IvoireConfigError(f"'{name}' in the settings must be a JSON object")
    return dict(s)
