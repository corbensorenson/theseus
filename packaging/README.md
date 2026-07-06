# Project Theseus Native Packaging

The source checkout now includes a native supervisor binary:

```bash
cargo build --release -p theseus-supervisor
```

That binary is the installer entrypoint for Windows, macOS, and Linux packaging.
It can run `doctor`, initialize runtime spillover paths, start the setup wizard,
and launch the dashboard plus Hive node.

## macOS

From the app, open the setup wizard and use **Installer USB**. From the CLI,
list candidate removable drives, then write the same universal bundle directly
to the selected USB:

```powershell
.\bin\theseus.cmd usb list
.\bin\theseus.cmd usb write --target E:\ --confirm-label YOUR_USB_LABEL --coordinator-url http://10.0.0.147:8791 --force
```

If no USB target is supplied, the writer still creates
`dist/universal-usb/ProjectTheseusUniversalUSB.zip` for manual copying.

Windows:

```powershell
.\Install Project Theseus Hive.cmd
```

The Windows installer creates a user-session tray operator by default. The tray
uses `assets/windows/theseus-hive.ico`, opens the Hive operator/chat PWA and
dashboard, can start/restart/stop local services, runs the Windows/CUDA doctor,
and watches local reports for training-improved, action-blocked,
teacher-needed, promotion-ready, and CUDA/resource-pressure notifications. Use
`-NoTray` only for headless workers.

Windows release signing:

```powershell
$env:THESEUS_WINDOWS_SIGNING_CERT_THUMBPRINT = "YOUR_CERT_THUMBPRINT"
powershell -ExecutionPolicy Bypass -File scripts\package_theseus_windows.ps1 -Force -BuildExe -Sign
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package_theseus_windows.ps1 `
  -Force -BuildExe -Sign `
  -SigningCertificatePath C:\secure\theseus-code-signing.pfx `
  -SigningCertificatePassword "local-password"
```

The signing helper is `scripts/sign_theseus_windows_artifacts.ps1`. It signs
the generated EXE with `signtool.exe`, signs PowerShell scripts with
Authenticode when a certificate is available, timestamps signatures, and emits
`dist/windows/windows-signing-report.json`. Private certificates and passwords
must remain local; never commit signing material. Authenticode signing reduces
Windows trust friction, but SmartScreen reputation still depends on the
publisher certificate and release history.

macOS:

```bash
open "Install Project Theseus Hive.command"
```

Linux:

```bash
sh install-from-usb.sh
```

The universal writer is [scripts/hive_usb_writer.py](../scripts/hive_usb_writer.py).
The legacy [scripts/build_macos_usb_bundle.py](../scripts/build_macos_usb_bundle.py)
entrypoint is only a compatibility wrapper around that shared source. The bundle
excludes Windows junctions, virtualenvs, build outputs, reports, checkpoints,
generated caches, ROM/game files, and local secret configs. It normalizes shell
scripts to LF line endings, preserves macOS/Linux executable bits in the zip,
and embeds an invite for the active Hive. Treat that zip/USB like a password
because it contains a join token.

Hive credential modes:

```powershell
# Join this machine's currently active private Hive.
.\bin\theseus.cmd usb write --target E:\ --confirm-label YOUR_USB_LABEL --hive-mode current --force

# Create and activate a new private Hive on this coordinator, then write that invite.
.\bin\theseus.cmd usb write --target E:\ --confirm-label YOUR_USB_LABEL --hive-mode new --new-hive-name "Workshop Hive" --force

# Configure target machines as public workers without carrying a private Hive token.
.\bin\theseus.cmd usb write --target E:\ --confirm-label YOUR_USB_LABEL --hive-mode public --public-gateway-url https://gateway.example --public-mode idle --force
```

Modern Windows, macOS, and Linux block silent USB autorun for security. The
writer still adds root launchers plus `autorun.inf` metadata so the correct
installer is the first obvious action, but a user gesture is required on a fresh
machine.

For direct source-folder Mac installs, run:

```bash
./scripts/install_theseus_hive_macos.sh --invite INVITE_FILE.json --coordinator-url http://COORDINATOR:8791
```

That copies the payload off the USB/source folder into
`~/Library/Application Support/Project Theseus Hive/app/current`, keeps mutable
runtime data in Application Support/Caches, creates `.app` launchers, bootstraps
the venv, installs MLX on Apple Silicon, applies the invite, and runs a Hive
probe.

To build a distributable macOS installer app/zip/pkg on a Mac:

```bash
THESEUS_MACOS_CODESIGN_IDENTITY="Developer ID Application: Example" \
THESEUS_MACOS_INSTALLER_IDENTITY="Developer ID Installer: Example" \
THESEUS_NOTARYTOOL_PROFILE="theseus-notary" \
./scripts/package_theseus_macos.sh
```

If no signing identity is supplied the script creates an ad-hoc signed local
artifact for private testing. Release signing and notarization require Apple
Developer ID credentials stored in the local keychain; private certificates,
Apple credentials, and package-signing keys must never be committed. Apple's
current command-line path is `codesign`, `notarytool`, and `stapler`.

Before installing on spare Intel or Apple-Silicon Macs, run the release gate:

```bash
python3 scripts/hive_macos_release_gate.py --execute
```

The DMG app now runs the macOS installer with safe soft auto-update, user
LaunchAgents, menu bar helper, and startup enabled by default. The gate confirms
the wrapper behavior, artifact hashes, universal intent, live `/mobile`,
private update catalog exposure, installer artifact exposure, local soft-update
convergence, and MLX capability on Apple Silicon. It also records the remaining
rollout blockers: physical Intel canary, Windows/Mac reachability, and
Developer ID notarization for non-technical distribution.
