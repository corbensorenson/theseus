import Foundation
import Security
import SwiftUI
import UserNotifications
import WatchConnectivity
import WatchKit

@MainActor
final class WatchHiveSettings: ObservableObject {
    @Published private(set) var nodeURLString: String
    @Published private(set) var endpointURLStrings: [String]
    @Published private(set) var hiveID: String
    @Published private(set) var token: String
    @Published private(set) var notificationsEnabled: Bool

    private let nodeURLKey = "theseus.watch.nodeURL"
    private let endpointURLsKey = "theseus.watch.endpointURLs"
    private let hiveIDKey = "theseus.watch.hiveID"
    private let tokenKey = "theseus.watch.token"
    private let notificationsEnabledKey = "theseus.watch.notificationsEnabled"

    init() {
        nodeURLString = UserDefaults.standard.string(forKey: nodeURLKey) ?? ""
        endpointURLStrings = UserDefaults.standard.stringArray(forKey: endpointURLsKey) ?? []
        hiveID = UserDefaults.standard.string(forKey: hiveIDKey) ?? ""
        token = WatchKeychainStore.read(service: tokenKey) ?? ""
        if UserDefaults.standard.object(forKey: notificationsEnabledKey) == nil {
            notificationsEnabled = true
        } else {
            notificationsEnabled = UserDefaults.standard.bool(forKey: notificationsEnabledKey)
        }
    }

    var isConfigured: Bool {
        !endpointBaseURLs.isEmpty && !token.isEmpty
    }

    var endpointBaseURLs: [URL] {
        normalizedEndpointStrings([nodeURLString] + endpointURLStrings).compactMap { normalizedBaseURL($0) }
    }

    func save(nodeURLString: String, endpointURLStrings: [String], hiveID: String, token: String) {
        let endpoints = normalizedEndpointStrings([nodeURLString] + endpointURLStrings)
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        let first = endpoints.first ?? nodeURLString.trimmingCharacters(in: .whitespacesAndNewlines)
        UserDefaults.standard.set(first, forKey: nodeURLKey)
        UserDefaults.standard.set(endpoints, forKey: endpointURLsKey)
        UserDefaults.standard.set(hiveID.trimmingCharacters(in: .whitespacesAndNewlines), forKey: hiveIDKey)
        if !trimmedToken.isEmpty {
            try? WatchKeychainStore.save(trimmedToken, service: tokenKey)
        }
        self.nodeURLString = first
        self.endpointURLStrings = endpoints
        self.hiveID = hiveID.trimmingCharacters(in: .whitespacesAndNewlines)
        self.token = trimmedToken
    }

    func clear() {
        UserDefaults.standard.removeObject(forKey: nodeURLKey)
        UserDefaults.standard.removeObject(forKey: endpointURLsKey)
        UserDefaults.standard.removeObject(forKey: hiveIDKey)
        WatchKeychainStore.delete(service: tokenKey)
        nodeURLString = ""
        endpointURLStrings = []
        hiveID = ""
        token = ""
    }

    func markActiveEndpoint(_ baseURL: URL, learnedEndpoints: [String] = []) {
        let normalized = normalizedBaseURL(baseURL.absoluteString)?.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/")) ?? baseURL.absoluteString
        let endpoints = normalizedEndpointStrings([normalized] + learnedEndpoints + endpointURLStrings)
        nodeURLString = normalized
        endpointURLStrings = endpoints
        UserDefaults.standard.set(normalized, forKey: nodeURLKey)
        UserDefaults.standard.set(endpoints, forKey: endpointURLsKey)
    }

    func setNotificationsEnabled(_ enabled: Bool) {
        notificationsEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: notificationsEnabledKey)
        if enabled {
            Task { await WatchHiveNotifier.requestAuthorization() }
        }
    }

    private func normalizedBaseURL(_ value: String) -> URL? {
        var text = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return nil }
        if !text.contains("://") {
            text = "http://" + text
        }
        while text.hasSuffix("/") {
            text.removeLast()
        }
        return URL(string: text)
    }

    private func normalizedEndpointStrings(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var out: [String] = []
        for value in values {
            guard let normalized = normalizedEndpointString(value), !seen.contains(normalized) else { continue }
            seen.insert(normalized)
            out.append(normalized)
        }
        return out
    }

    private func normalizedEndpointString(_ value: String) -> String? {
        var text = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty || text.hasPrefix("theseushive://") { return nil }
        if !text.contains("://") {
            text = "http://" + text
        }
        guard var components = URLComponents(string: text), components.scheme != nil, components.host != nil else { return nil }
        if components.path == "/mobile" || components.path == "/m" || components.path == "/operator" {
            components.path = ""
        }
        components.queryItems = nil
        components.fragment = nil
        return components.url?.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }
}

struct WatchHiveStatus {
    enum State {
        case offline
        case running
        case connected
    }

    var state: State = .offline
    var title: String = "No Hive"
    var detail: String = "Open the iPhone app to send a profile."
    var peerCount: Int = 0
    var version: String = "--"
    var utilization: String = "--"
    var mlx: String = "MLX --"
    var activeEndpoint: String = ""
    var lastUpdated = Date.distantPast

    var color: Color {
        switch state {
        case .offline: return .red
        case .running: return .yellow
        case .connected: return .green
        }
    }
}

@MainActor
final class WatchHiveMonitor: ObservableObject {
    @Published private(set) var status = WatchHiveStatus()
    @Published private(set) var lastMessage = ""
    @Published private(set) var isBusy = false

    func refresh(settings: WatchHiveSettings) async {
        guard settings.isConfigured else {
            status = WatchHiveStatus()
            return
        }
        isBusy = true
        defer { isBusy = false }
        let previousStatus = status
        var sawRejected = false
        for base in settings.endpointBaseURLs {
            do {
                let result = try await fetchOperatorStatus(base: base, token: settings.token)
                settings.markActiveEndpoint(base, learnedEndpoints: result.learnedEndpoints)
                let wasOffline = status.state == .offline
                status = result.status
                if wasOffline {
                    WKInterfaceDevice.current().play(.success)
                }
                await notifyStatusChange(previous: previousStatus, next: result.status, settings: settings)
                return
            } catch WatchHiveError.rejected {
                sawRejected = true
            } catch {
            }
            do {
                if let result = try await fetchRelayStatus(base: base, hiveID: settings.hiveID, token: settings.token) {
                    settings.markActiveEndpoint(base, learnedEndpoints: result.learnedEndpoints)
                    let wasOffline = status.state == .offline
                    status = result.status
                    if wasOffline {
                        WKInterfaceDevice.current().play(.success)
                    }
                    await notifyStatusChange(previous: previousStatus, next: result.status, settings: settings)
                    return
                }
            } catch WatchHiveError.rejected {
                sawRejected = true
            } catch {
            }
        }
        let nextStatus = WatchHiveStatus(
            state: .offline,
            title: sawRejected ? "Token Rejected" : "Hive Offline",
            detail: sawRejected ? "Check the iPhone profile." : "No saved endpoint answered.",
            peerCount: 0,
            version: "--",
            utilization: "--",
            mlx: "MLX --",
            activeEndpoint: "",
            lastUpdated: Date()
        )
        status = nextStatus
        WKInterfaceDevice.current().play(.failure)
        await notifyStatusChange(previous: previousStatus, next: nextStatus, settings: settings)
    }

    func sendChat(_ text: String, settings: WatchHiveSettings) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        await post(path: "api/hive/operator/chat", payload: ["message": trimmed], settings: settings)
    }

    func checkpoint(settings: WatchHiveSettings) async {
        await post(path: "api/hive/operator/task", payload: ["kind": "checkpoint_chat", "payload": ["source": "watch"]], settings: settings)
    }

    private func post(path: String, payload: [String: Any], settings: WatchHiveSettings) async {
        guard settings.isConfigured else { return }
        isBusy = true
        defer { isBusy = false }
        var sawRejected = false
        for base in settings.endpointBaseURLs {
            do {
                try await postOnce(base: base, path: path, payload: payload, token: settings.token)
                settings.markActiveEndpoint(base)
                lastMessage = "Queued."
                WKInterfaceDevice.current().play(.success)
                if settings.notificationsEnabled {
                    await WatchHiveNotifier.notify(
                        title: "Hive Command Queued",
                        body: "Theseus accepted the Watch command.",
                        identifier: "theseus-watch-command-\(Int(Date().timeIntervalSince1970))"
                    )
                }
                await refresh(settings: settings)
                return
            } catch WatchHiveError.rejected {
                sawRejected = true
            } catch {
                continue
            }
        }
        lastMessage = sawRejected ? "Hive rejected command." : "Command failed."
        WKInterfaceDevice.current().play(.failure)
        if settings.notificationsEnabled {
            await WatchHiveNotifier.notify(
                title: "Hive Command Failed",
                body: lastMessage,
                identifier: "theseus-watch-command-failed-\(Int(Date().timeIntervalSince1970))"
            )
        }
    }

    private func postOnce(base: URL, path: String, payload: [String: Any], token: String) async throws {
        let url = endpoint(base: base, path: path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 10
        request.setValue(token, forHTTPHeaderField: "X-Theseus-Hive-Secret")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw WatchHiveError.unreadable }
        if http.statusCode == 401 || http.statusCode == 403 {
            throw WatchHiveError.rejected
        }
        guard (200..<300).contains(http.statusCode) else { throw WatchHiveError.unreadable }
    }

    private func fetchOperatorStatus(base: URL, token: String) async throws -> WatchProbeResult {
        let url = endpoint(base: base, path: "api/hive/operator/status")
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        request.setValue(token, forHTTPHeaderField: "X-Theseus-Hive-Secret")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw WatchHiveError.unreadable }
        if http.statusCode == 401 || http.statusCode == 403 {
            throw WatchHiveError.rejected
        }
        guard (200..<300).contains(http.statusCode) else { throw WatchHiveError.unreadable }
        return parseOperatorStatus(data, base: base)
    }

    private func fetchRelayStatus(base: URL, hiveID: String, token: String) async throws -> WatchProbeResult? {
        guard !hiveID.isEmpty else { return nil }
        let url = endpoint(base: base, path: "api/hive/relay/peers", queryItems: [URLQueryItem(name: "hive_id", value: hiveID)])
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        request.setValue(token, forHTTPHeaderField: "X-Theseus-Hive-Secret")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw WatchHiveError.unreadable }
        if http.statusCode == 401 || http.statusCode == 403 {
            throw WatchHiveError.rejected
        }
        guard (200..<300).contains(http.statusCode) else { return nil }
        return parseRelayStatus(data, base: base)
    }

    private func parseOperatorStatus(_ data: Data, base: URL) -> WatchProbeResult {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return WatchProbeResult(status: WatchHiveStatus(title: "Unreadable", detail: "Unexpected Hive response."), learnedEndpoints: [])
        }
        let hive = object["hive"] as? [String: Any] ?? [:]
        let local = hive["local_node"] as? [String: Any] ?? [:]
        let peerCount = hive["peer_count"] as? Int ?? 0
        let nodeName = local["node_name"] as? String ?? "Theseus Hive"
        let utilization = object["utilization"] as? [String: Any] ?? [:]
        let accelerators = object["accelerators"] as? [String: Any] ?? [:]
        let appleMlx = accelerators["apple_mlx"] as? [String: Any] ?? [:]
        let version = ((object["version"] as? [String: Any])?["local_version_id"] as? String)
            ?? ((local["hive_version"] as? [String: Any])?["local_version_id"] as? String)
            ?? "--"
        let trigger = utilization["trigger_state"] as? String ?? "--"
        let mlxReady = appleMlx["available"] as? Bool ?? false
        let mlxCount = appleMlx["node_count"] as? Int ?? 0
        let state: WatchHiveStatus.State = peerCount > 0 ? .connected : .running
        let shortVersion = version.replacingOccurrences(of: "hive-", with: "")
        return WatchProbeResult(
            status: WatchHiveStatus(
                state: state,
                title: state == .connected ? "Hive Connected" : "Hive Running",
                detail: peerCount > 0 ? "\(nodeName) sees \(peerCount) peer(s)." : nodeName,
                peerCount: peerCount,
                version: shortVersion == "--" ? "--" : String(shortVersion.prefix(8)),
                utilization: trigger,
                mlx: mlxReady ? "MLX \(max(mlxCount, 1))" : "MLX --",
                activeEndpoint: base.host ?? base.absoluteString,
                lastUpdated: Date()
            ),
            learnedEndpoints: learnedEndpoints(from: object)
        )
    }

    private func parseRelayStatus(_ data: Data, base: URL) -> WatchProbeResult {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return WatchProbeResult(status: WatchHiveStatus(title: "Unreadable Relay", detail: "Unexpected relay response."), learnedEndpoints: [])
        }
        let peers = object["peers"] as? [[String: Any]] ?? []
        let peerCount = peers.count
        let state: WatchHiveStatus.State = peerCount > 0 ? .connected : .running
        return WatchProbeResult(
            status: WatchHiveStatus(
                state: state,
                title: peerCount > 0 ? "Relay Connected" : "Relay Online",
                detail: peerCount > 0 ? "\(peerCount) node(s) reachable through \(base.host ?? "relay")." : "Relay is online; waiting for nodes to poll.",
                peerCount: peerCount,
                version: "--",
                utilization: "relay",
                mlx: "MLX via peers",
                activeEndpoint: base.host ?? base.absoluteString,
                lastUpdated: Date()
            ),
            learnedEndpoints: learnedEndpoints(fromPeers: peers)
        )
    }

    private func learnedEndpoints(from object: [String: Any]) -> [String] {
        var raw: [String] = []
        if let hive = object["hive"] as? [String: Any] {
            if let local = hive["local_node"] as? [String: Any] {
                appendString(local["api_url"], to: &raw)
                appendString(local["relay_url"], to: &raw)
            }
            if let peers = hive["peers"] as? [[String: Any]] {
                raw += learnedEndpoints(fromPeers: peers)
            }
        }
        if let roaming = object["roaming"] as? [String: Any], let endpoints = roaming["endpoints"] as? [[String: Any]] {
            for endpoint in endpoints {
                appendString(endpoint["url"], to: &raw)
            }
        }
        return raw
    }

    private func learnedEndpoints(fromPeers peers: [[String: Any]]) -> [String] {
        var raw: [String] = []
        for peer in peers {
            appendString(peer["api_url"], to: &raw)
            appendString(peer["relay_url"], to: &raw)
        }
        return raw
    }

    private func appendString(_ value: Any?, to raw: inout [String]) {
        if let text = value as? String, !text.isEmpty {
            raw.append(text)
        }
    }

    private func notifyStatusChange(previous: WatchHiveStatus, next: WatchHiveStatus, settings: WatchHiveSettings) async {
        guard settings.notificationsEnabled else { return }
        if previous.state != .offline && next.state == .offline {
            await WatchHiveNotifier.notify(
                title: next.title,
                body: next.detail,
                identifier: "theseus-watch-hive-offline"
            )
        } else if previous.state == .running && next.state == .connected {
            await WatchHiveNotifier.notify(
                title: "Hive Connected",
                body: next.detail,
                identifier: "theseus-watch-hive-connected"
            )
        }
    }

    private func endpoint(base: URL, path: String, queryItems: [URLQueryItem] = []) -> URL {
        var components = URLComponents(url: base, resolvingAgainstBaseURL: false)
        components?.path = "/" + path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        components?.queryItems = queryItems.isEmpty ? nil : queryItems
        return components?.url ?? base.appendingPathComponent(path)
    }
}

struct WatchProbeResult {
    let status: WatchHiveStatus
    let learnedEndpoints: [String]
}

enum WatchHiveError: Error {
    case rejected
    case unreadable
}

enum WatchHiveNotifier {
    static func requestAuthorization() async {
        _ = try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound])
    }

    static func notify(title: String, body: String, identifier: String) async {
        let center = UNUserNotificationCenter.current()
        let settings = await center.notificationSettings()
        switch settings.authorizationStatus {
        case .authorized, .provisional:
            break
        case .notDetermined:
            await requestAuthorization()
        default:
            return
        }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        try? await center.add(UNNotificationRequest(identifier: identifier, content: content, trigger: nil))
    }
}

@MainActor
final class WatchProfileReceiver: NSObject, ObservableObject {
    @Published private(set) var lastProfileUTC = ""
    private weak var settings: WatchHiveSettings?
    private let seenNotificationKey = "theseus.watch.seenNotificationIDs"

    func start(settings: WatchHiveSettings) {
        self.settings = settings
        guard WCSession.isSupported() else { return }
        let session = WCSession.default
        if session.delegate == nil {
            session.delegate = self
            session.activate()
        }
        if !session.receivedApplicationContext.isEmpty {
            apply(session.receivedApplicationContext)
        }
    }

    private func apply(_ payload: [String: Any]) {
        if payload["policy"] as? String == "project_theseus_watch_notification_v0" {
            applyNotification(payload)
            return
        }
        guard payload["policy"] as? String == "project_theseus_watch_profile_v0" else { return }
        let nodeURL = payload["node_url"] as? String ?? ""
        let endpoints = payload["endpoint_urls"] as? [String] ?? []
        let hiveID = payload["hive_id"] as? String ?? ""
        let token = payload["operator_token"] as? String ?? ""
        guard !nodeURL.isEmpty, !token.isEmpty else { return }
        settings?.save(nodeURLString: nodeURL, endpointURLStrings: endpoints, hiveID: hiveID, token: token)
        if settings?.notificationsEnabled == true {
            Task { await WatchHiveNotifier.requestAuthorization() }
        }
        lastProfileUTC = payload["updated_utc"] as? String ?? ISO8601DateFormatter().string(from: Date())
    }

    private func applyNotification(_ payload: [String: Any]) {
        guard settings?.notificationsEnabled == true else { return }
        guard let id = payload["id"] as? String, !id.isEmpty else { return }
        var seen = Set(UserDefaults.standard.stringArray(forKey: seenNotificationKey) ?? [])
        guard !seen.contains(id) else { return }
        seen.insert(id)
        let kept = Array(seen).suffix(300)
        UserDefaults.standard.set(Array(kept), forKey: seenNotificationKey)
        let title = payload["title"] as? String ?? "Theseus Hive"
        let body = payload["body"] as? String ?? ""
        let identifier = "theseus-watch-relay-\(id)"
        Task {
            await WatchHiveNotifier.notify(title: title, body: body, identifier: identifier)
            WKInterfaceDevice.current().play(.notification)
        }
    }
}

extension WatchProfileReceiver: WCSessionDelegate {
    nonisolated func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {}

    nonisolated func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String: Any]) {
        Task { @MainActor in
            self.apply(applicationContext)
        }
    }

    nonisolated func session(_ session: WCSession, didReceiveUserInfo userInfo: [String: Any] = [:]) {
        Task { @MainActor in
            self.apply(userInfo)
        }
    }

    nonisolated func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
        Task { @MainActor in
            self.apply(message)
        }
    }
}

enum WatchKeychainStore {
    static func save(_ value: String, service: String) throws {
        let data = Data(value.utf8)
        delete(service: service)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: "default",
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else { throw WatchKeychainError.unhandled(status) }
    }

    static func read(service: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: "default",
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func delete(service: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: "default"
        ]
        SecItemDelete(query as CFDictionary)
    }
}

enum WatchKeychainError: Error {
    case unhandled(OSStatus)
}
