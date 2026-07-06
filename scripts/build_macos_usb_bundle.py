"""Compatibility wrapper for the universal Hive USB writer."""

from __future__ import annotations

from pathlib import Path

import hive_usb_writer


if __name__ == "__main__":
    default_out = hive_usb_writer.ROOT / "dist" / "macos-usb" / "ProjectTheseusMacUSB"
    raise SystemExit(hive_usb_writer.main(default_out=Path(default_out)))
