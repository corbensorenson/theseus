# Theseus Hive Android App

This is the native Android operator shell for Project Theseus Hive. It mirrors
the iPhone app shape: a native connection/settings layer around the trusted
`/mobile` operator surface.

Current scope:

- Android 8.0 / API 26 or newer.
- Private LAN, hotspot, private tunnel, or authenticated relay operator use.
- Manual node URL + invite/operator token setup.
- Paste/import private Hive invite JSON or per-user operator invite JSON from
  `theseus hive add-user`.
- Native connection status against `/api/hive/operator/status`.
- WebView operator/chat/task UI against `/mobile`.
- Encrypted invite token storage through Android Keystore.
- Emulator-friendly Windows host access through `http://10.0.2.2:8791`.

Build from Windows after installing Android tooling:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_theseus_android.ps1 -AcceptLicenses
powershell -ExecutionPolicy Bypass -File scripts\build_theseus_android.ps1
```

The setup script installs Android command-line tools, SDK packages, Gradle, and
Temurin JDK 17 under `D:\ProjectTheseus` by default. It writes:

```text
D:\ProjectTheseus\android-sdk
D:\ProjectTheseus\android-tools
D:\ProjectTheseus\android-tools\jdk-17
D:\ProjectTheseus\android-avd
```

Emulator smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_theseus_android_emulator.ps1
powershell -ExecutionPolicy Bypass -File scripts\build_theseus_android.ps1 -InstallDebug
```

The emulator and AVD can live entirely on `D:`, but booting an x86_64 Android
image still requires Windows hardware acceleration. If
`start_theseus_android_emulator.ps1` reports
`android_emulator_acceleration_unavailable`, enable CPU virtualization and
install Android Emulator Hypervisor Driver or Windows Hypervisor Platform, then
rerun the emulator script.

Inside the emulator, use:

```text
http://10.0.2.2:8791
```

for the Windows host Hive node. On a real Android phone, use the node LAN,
hotspot, VPN, or relay URL from the private invite.

Launch override smoke:

```powershell
adb shell am start `
  -n local.projecttheseus.hive.android/local.projecttheseus.hive.MainActivity `
  -e theseus_node_url http://10.0.2.2:8791 `
  -e theseus_token YOUR_PRIVATE_HIVE_TOKEN
```

The app is an operator client, not a training worker. It queues only registered
Hive operator tasks through the same private task allowlist as the PWA.
