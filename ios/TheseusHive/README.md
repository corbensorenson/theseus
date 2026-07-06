# Theseus Hive iPhone And Apple Watch Apps

This is the native Apple operator shell for Project Theseus Hive. The iPhone
app wraps the trusted `/mobile` operator surface in SwiftUI, stores the invite
token in the iOS Keychain, and keeps a native connection banner above the web
operator. The Apple Watch app is a paired native companion for at-a-glance Hive
health, text/dictation commands, haptics, and local alerts.

Current scope:

- iOS 16.0 or newer, including iPhone X-class devices.
- watchOS 9.0 or newer for the native `TheseusHiveWatch` companion.
- Private LAN, hotspot, self-hosted WireGuard/private tunnel, or authenticated
  relay Hive operator control.
- Manual node URL + token setup.
- Paste/import the private Hive invite JSON, the one-step bootstrap bundle from
  `theseus bootstrap`, scan the setup wizard's `theseushive://join` QR profile
  with the native camera scanner, or open the same roaming profile deep link.
- Per-user operator invite JSON from `theseus hive add-user`, so a family can
  give each phone its own revocable role instead of sharing the machine join
  secret everywhere.
- Saved endpoint candidates for home Wi-Fi, hotspot, Mac Bonjour `.local`,
  private tunnel, and relay fallback. The app promotes the first reachable
  endpoint and persists it as the next last-good candidate.
- Native connection status against `/api/hive/operator/status`.
- WKWebView operator/chat/task UI against `/mobile`.
- Native status strip for peer count, always-active state, blocked nodes, and
  Apple MLX readiness before the web operator finishes loading.
- Native wrapper passes invite tokens through a URL fragment and the web
  operator scrubs the URL after import, so private tokens are not sent as the
  page request URL.
- Distributed Training panel from `/mobile`, including active arm owners,
  current Hive training round, and the bounded `training_orchestrate` quick
  action.
- Storage panel support through `/mobile` for browsing configured Hive storage
  shares on the connected node or trusted peers and previewing files such as
  photos without remote-controlling the desktop.
- Apple MLX readiness and bounded MLX quick tasks exposed through the shared
  operator contract when an M-series Mac node advertises `mlx_apple`.
- Intel Mac Hive nodes show up as CPU/storage/operator-capable peers; MLX quick
  tasks stay routed to Apple Silicon Macs or other nodes that advertise an MLX
  backend.
- The iPhone app sends the current node URL, endpoint candidates, Hive ID, and
  operator token to the Watch through WatchConnectivity. The Watch stores the
  token in its own Keychain and can be configured manually if pairing sync is
  unavailable.
- The Watch app shows live health, peer count, Hive version, always-active work
  state, MLX availability, active endpoint, text/dictation chat, a checkpoint
  quick action, an overnight summary prompt, local notifications, and haptics.
  It now understands both direct node status and relay peer status, so it can
  remain useful when the phone/watch profile is using a roaming relay path. It
  only queues bounded operator tasks; it is not a training worker.
- The iPhone app polls `/api/hive/operator/notifications` while active, shows
  local alerts for interruptive Hive events, acknowledges delivered events, and
  relays the same alerts to the paired Watch through WatchConnectivity.

Build locally:

```bash
./scripts/build_theseus_ios.sh --validate

IOS_SIM_ID="$(xcrun simctl list devices available | awk -F'[()]' '/iPhone 16 Pro/ {print $2; exit}')"
xcodebuild -project ios/TheseusHive/TheseusHive.xcodeproj -scheme TheseusHive -destination "platform=iOS Simulator,id=$IOS_SIM_ID" CODE_SIGNING_ALLOWED=NO build
xcodebuild -project ios/TheseusHive/TheseusHive.xcodeproj -scheme TheseusHiveWatch CODE_SIGNING_ALLOWED=NO build
```

`build_theseus_ios.sh --validate` writes `reports/ios/build_status.json` and
logs under `reports/ios/logs/`. It validates the watchOS simulator target and
the unsigned iPhone+Watch device build without requiring Codex on the target
phone. The combined iPhone simulator scheme can require a matching paired
watchOS simulator runtime; use the validation script as the source-of-truth
check when that runtime is not installed.

The iPhone scheme embeds the Watch content, so a signed real-device install can
carry both apps to a paired iPhone and Apple Watch. Use a concrete iOS simulator
destination when building the iPhone scheme; generic iOS destinations can ask
Xcode for a newer watchOS runtime than is installed.

Signing and TestFlight archive:

```bash
./scripts/build_theseus_ios.sh \
  --archive \
  --team YOUR_APPLE_TEAM_ID \
  --app-bundle-id com.yourname.theseushive \
  --watch-bundle-id com.yourname.theseushive.watch \
  --allow-provisioning-updates
```

The archive is written to `dist/ios/TheseusHive.xcarchive` by default and can
be uploaded from Xcode Organizer for TestFlight. The default bundle IDs are
`com.corbensorenson.theseushive` and `com.corbensorenson.theseushive.watch`;
change them in Xcode or pass the flags above before registering App Store
Connect identifiers.

Simulator smoke test:

```bash
xcrun simctl boot "iPhone 16 Pro"
xcrun simctl install booted ~/Library/Developer/Xcode/DerivedData/TheseusHive-*/Build/Products/Debug-iphonesimulator/TheseusHive.app
SIMCTL_CHILD_THESEUS_HIVE_IOS_NODE_URL=http://127.0.0.1:8791 \
SIMCTL_CHILD_THESEUS_HIVE_IOS_TOKEN=local-smoke \
  xcrun simctl launch --terminate-running-process booted local.project-theseus.hive.ios
```

For device install, open the project in Xcode, set a signing team for both
targets, confirm the iPhone and Watch bundle IDs match the values you registered,
and run the `TheseusHive` scheme on the iPhone while it is on the same trusted
network, hotspot, self-hosted WireGuard/private tunnel, or authenticated relay
path as a Hive node. Xcode should install the embedded Watch app on a paired
Watch; the `TheseusHiveWatch` scheme can also be run directly on a watchOS 9+
simulator or paired Watch. The apps do not require a paid mesh service.

Watch notifications currently depend on the iPhone or Watch app being active
often enough to poll/receive the Hive notification feed. True background server
push should be added later through APNs so overnight Hive events can wake the
Watch without foreground polling.

Roaming profile:

```bash
theseus remote mobile-profile
theseus bootstrap --out reports/hive_join_bundle.json --qr-out reports/hive_join_profile_qr.svg
```

This writes `reports/hive_mobile_roaming_profile.json`, including all known
node and relay candidates plus a private app deep link. The profile includes
Bonjour service metadata, `.local` Mac candidates, endpoint priorities, and a
handoff policy: last-good first, Bonjour/LAN next, private tunnel next, HTTPS
relay last. `theseus bootstrap` writes the broader machine/phone join bundle
and QR used by non-Codex installs.
In the iPhone app, open settings and use **Scan Join QR** or **Import Invite
JSON**; both paths save the profile and sync it to the paired Watch. The
`/mobile` operator also has a Roaming panel backed by
`/api/hive/operator/roaming-profile`; it can build an import link from the
currently supplied operator token without revealing the machine join token.
Treat token-bearing files and QR codes like invite tokens. Cellular/offsite use
still needs one reachable free path: self-hosted WireGuard/private tunnel or an
HTTPS-protected Hive relay. The app can pick among saved paths automatically,
but it cannot reach a private home network through NAT unless a tunnel or relay
exists.
