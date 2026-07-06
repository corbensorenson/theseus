import Foundation
import SwiftUI

struct HiveInvite {
    let hiveID: String
    let coordinatorURL: String
    let endpointURLs: [String]
    let joinToken: String
}

@MainActor
final class HiveSettings: ObservableObject {
    @Published private(set) var nodeURLString: String
    @Published private(set) var endpointURLStrings: [String]
    @Published private(set) var hiveID: String
    @Published private(set) var token: String
    @Published private(set) var notificationsEnabled: Bool

    private let nodeURLKey = "theseus.hive.nodeURL"
    private let endpointURLsKey = "theseus.hive.endpointURLs"
    private let hiveIDKey = "theseus.hive.hiveID"
    private let tokenKey = "theseus.hive.token"
    private let notificationsEnabledKey = "theseus.hive.notificationsEnabled"

    init() {
        let launch = HiveLaunchOverrides.current
        let savedEndpoints = UserDefaults.standard.stringArray(forKey: endpointURLsKey) ?? []
        let launchEndpoint = launch.nodeURL.map { [$0] } ?? []
        let initialEndpoints = Self.normalizedEndpointStrings(launchEndpoint + savedEndpoints)
        endpointURLStrings = initialEndpoints
        nodeURLString = launch.nodeURL ?? UserDefaults.standard.string(forKey: nodeURLKey) ?? initialEndpoints.first ?? ""
        hiveID = UserDefaults.standard.string(forKey: hiveIDKey) ?? ""
        token = launch.token ?? KeychainStore.read(service: tokenKey) ?? ""
        if UserDefaults.standard.object(forKey: notificationsEnabledKey) == nil {
            notificationsEnabled = true
        } else {
            notificationsEnabled = UserDefaults.standard.bool(forKey: notificationsEnabledKey)
        }
    }

    var isConfigured: Bool {
        !endpointBaseURLs.isEmpty && !token.isEmpty
    }

    var mobileURL: URL? {
        guard let base = Self.normalizedBaseURL(nodeURLString) else { return nil }
        return url(base: base, path: "mobile")
    }

    var operatorStatusURL: URL? {
        guard let base = Self.normalizedBaseURL(nodeURLString) else { return nil }
        return url(base: base, path: "api/hive/operator/status")
    }

    var endpointBaseURLs: [URL] {
        Self.normalizedEndpointStrings([nodeURLString] + endpointURLStrings).compactMap { Self.normalizedBaseURL($0) }
    }

    var endpointURLText: String {
        endpointURLStrings.joined(separator: "\n")
    }

    func save(nodeURLString: String, token: String) throws {
        try save(nodeURLString: nodeURLString, endpointURLStrings: [nodeURLString], hiveID: hiveID, token: token)
    }

    func save(nodeURLString: String, endpointURLStrings: [String], hiveID: String, token: String) throws {
        let endpoints = Self.normalizedEndpointStrings([nodeURLString] + endpointURLStrings)
        guard let first = endpoints.first, Self.normalizedBaseURL(first) != nil else {
            throw HiveSettingsError.invalidURL
        }
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedToken.isEmpty else {
            throw HiveSettingsError.missingToken
        }
        let normalized = Self.normalizedBaseURL(first)!.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        UserDefaults.standard.set(normalized, forKey: nodeURLKey)
        UserDefaults.standard.set(endpoints, forKey: endpointURLsKey)
        UserDefaults.standard.set(hiveID.trimmingCharacters(in: .whitespacesAndNewlines), forKey: hiveIDKey)
        try KeychainStore.save(trimmedToken, service: tokenKey)
        self.nodeURLString = normalized
        self.endpointURLStrings = endpoints
        self.hiveID = hiveID.trimmingCharacters(in: .whitespacesAndNewlines)
        self.token = trimmedToken
    }

    func clear() {
        UserDefaults.standard.removeObject(forKey: nodeURLKey)
        UserDefaults.standard.removeObject(forKey: endpointURLsKey)
        UserDefaults.standard.removeObject(forKey: hiveIDKey)
        KeychainStore.delete(service: tokenKey)
        nodeURLString = ""
        endpointURLStrings = []
        hiveID = ""
        token = ""
    }

    func markActiveEndpoint(_ baseURL: URL, learnedEndpoints: [String] = []) {
        let normalized = Self.normalizedBaseURL(baseURL.absoluteString)?.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/")) ?? baseURL.absoluteString
        let endpoints = Self.normalizedEndpointStrings([normalized] + learnedEndpoints + endpointURLStrings)
        nodeURLString = normalized
        endpointURLStrings = endpoints
        UserDefaults.standard.set(normalized, forKey: nodeURLKey)
        UserDefaults.standard.set(endpoints, forKey: endpointURLsKey)
    }

    func setNotificationsEnabled(_ enabled: Bool) {
        notificationsEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: notificationsEnabledKey)
    }

    func parseInvite(_ raw: String) throws -> HiveInvite {
        guard let data = raw.data(using: .utf8),
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw HiveSettingsError.invalidInvite
        }
        let hiveID = object["hive_id"] as? String ?? ""
        let joinToken = object["join_token"] as? String ?? object["operator_token"] as? String ?? object["token"] as? String ?? ""
        let endpoints = endpointCandidates(from: object)
        guard !hiveID.isEmpty, let coordinatorURL = endpoints.first, !joinToken.isEmpty else {
            throw HiveSettingsError.invalidInvite
        }
        return HiveInvite(hiveID: hiveID, coordinatorURL: coordinatorURL, endpointURLs: endpoints, joinToken: joinToken)
    }

    func applyJoinURL(_ url: URL) throws {
        guard url.scheme == "theseushive" else {
            throw HiveSettingsError.invalidInvite
        }
        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        let profile = components?.queryItems?.first(where: { $0.name == "profile" })?.value ?? ""
        if !profile.isEmpty {
            let json = try decodeBase64URLString(profile)
            let invite = try parseInvite(json)
            try save(nodeURLString: invite.coordinatorURL, endpointURLStrings: invite.endpointURLs, hiveID: invite.hiveID, token: invite.joinToken)
            return
        }
        let endpoint = components?.queryItems?.first(where: { $0.name == "url" || $0.name == "u" })?.value ?? ""
        let token = components?.queryItems?.first(where: { $0.name == "token" || $0.name == "t" })?.value ?? ""
        let hiveID = components?.queryItems?.first(where: { $0.name == "hive_id" || $0.name == "h" })?.value ?? ""
        guard Self.normalizedBaseURL(endpoint) != nil, !token.isEmpty else {
            throw HiveSettingsError.invalidInvite
        }
        try save(nodeURLString: endpoint, endpointURLStrings: [endpoint], hiveID: hiveID, token: token)
    }

    private static func normalizedBaseURL(_ value: String) -> URL? {
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

    private func url(base: URL, path: String, queryItems: [URLQueryItem] = []) -> URL {
        var components = URLComponents(url: base, resolvingAgainstBaseURL: false)
        components?.path = "/" + path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        components?.queryItems = queryItems.isEmpty ? nil : queryItems
        return components?.url ?? base.appendingPathComponent(path)
    }

    private func endpointCandidates(from object: [String: Any]) -> [String] {
        var raw: [String] = []
        appendString(object["coordinator_url"], to: &raw)
        appendString(object["relay_url"], to: &raw)
        appendStringArray(object["coordinator_urls"], to: &raw)
        appendStringArray(object["node_urls"], to: &raw)
        appendStringArray(object["relay_urls"], to: &raw)
        appendStringArray(object["operator_urls"], to: &raw)
        if let install = object["install"] as? [String: Any] {
            appendStringArray(install["phone"], to: &raw)
        }
        if let roaming = object["roaming"] as? [String: Any] {
            appendStringArray(roaming["coordinator_urls"], to: &raw)
            appendStringArray(roaming["node_urls"], to: &raw)
            appendStringArray(roaming["relay_urls"], to: &raw)
            if let endpoints = roaming["endpoints"] as? [[String: Any]] {
                for endpoint in endpoints {
                    appendString(endpoint["url"], to: &raw)
                    appendString(endpoint["operator_url"], to: &raw)
                }
            }
        }
        return Self.normalizedEndpointStrings(raw)
    }

    private func appendString(_ value: Any?, to raw: inout [String]) {
        if let text = value as? String {
            raw.append(text)
        }
    }

    private func appendStringArray(_ value: Any?, to raw: inout [String]) {
        guard let rows = value as? [Any] else { return }
        for row in rows {
            if let text = row as? String {
                raw.append(text)
            } else if let object = row as? [String: Any] {
                appendString(object["url"], to: &raw)
                appendString(object["operator_url"], to: &raw)
            }
        }
    }

    private static func normalizedEndpointStrings(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var out: [String] = []
        for value in values {
            guard let normalized = normalizedEndpointString(value), !seen.contains(normalized) else {
                continue
            }
            seen.insert(normalized)
            out.append(normalized)
        }
        return out
    }

    private static func normalizedEndpointString(_ value: String) -> String? {
        var text = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return nil }
        if text.hasPrefix("theseushive://") { return nil }
        if !text.contains("://") {
            text = "http://" + text
        }
        guard var components = URLComponents(string: text), components.scheme != nil, components.host != nil else {
            return nil
        }
        if components.path == "/mobile" || components.path == "/m" || components.path == "/operator" {
            components.path = ""
        }
        components.queryItems = nil
        components.fragment = nil
        return components.url?.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    private func decodeBase64URLString(_ value: String) throws -> String {
        var text = value.replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        let remainder = text.count % 4
        if remainder > 0 {
            text += String(repeating: "=", count: 4 - remainder)
        }
        guard let data = Data(base64Encoded: text), let json = String(data: data, encoding: .utf8) else {
            throw HiveSettingsError.invalidInvite
        }
        return json
    }
}

private struct HiveLaunchOverrides {
    let nodeURL: String?
    let token: String?

    static var current: HiveLaunchOverrides {
        let env = ProcessInfo.processInfo.environment
        return HiveLaunchOverrides(
            nodeURL: env["THESEUS_HIVE_IOS_NODE_URL"] ?? argumentValue("--theseus-node-url"),
            token: env["THESEUS_HIVE_IOS_TOKEN"] ?? argumentValue("--theseus-token")
        )
    }

    private static func argumentValue(_ name: String) -> String? {
        let args = ProcessInfo.processInfo.arguments
        for index in args.indices {
            let arg = args[index]
            if arg == name, args.indices.contains(index + 1) {
                return args[index + 1]
            }
            if arg.hasPrefix(name + "=") {
                return String(arg.dropFirst(name.count + 1))
            }
        }
        return nil
    }
}

enum HiveSettingsError: LocalizedError {
    case invalidURL
    case missingToken
    case invalidInvite

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Enter a valid Hive node URL, for example http://10.0.0.251:8791."
        case .missingToken:
            return "Enter the private Hive invite token."
        case .invalidInvite:
            return "The invite JSON must include hive_id, an endpoint URL, and a Hive or operator token."
        }
    }
}

struct HiveStatus {
    enum State {
        case offline
        case running
        case connected
    }

    var state: State = .offline
    var title: String = "Not Connected"
    var subtitle: String = "Add a Hive node URL and invite token."
    var peerCount: Int = 0
    var utilizationState: String = "--"
    var blockedNodeCount: Int = 0
    var mlxSummary: String = "MLX --"
    var activeEndpoint: String = ""

    var color: Color {
        switch state {
        case .offline: return .red
        case .running: return .yellow
        case .connected: return .green
        }
    }
}

@MainActor
final class HiveMonitor: ObservableObject {
    @Published private(set) var status = HiveStatus()

    func refresh(settings: HiveSettings) async {
        let candidates = settings.endpointBaseURLs
        guard !candidates.isEmpty else {
            status = HiveStatus()
            return
        }
        var rejected = false
        var lastError = ""
        for base in candidates {
            do {
                if let result = try await probeNode(base: base, token: settings.token) {
                    settings.markActiveEndpoint(base, learnedEndpoints: result.learnedEndpoints)
                    status = result.status
                    return
                }
                if let result = try await probeRelay(base: base, hiveID: settings.hiveID, token: settings.token) {
                    settings.markActiveEndpoint(base, learnedEndpoints: result.learnedEndpoints)
                    status = result.status
                    return
                }
            } catch HiveProbeError.rejected {
                rejected = true
            } catch {
                lastError = error.localizedDescription
            }
        }
        if rejected {
            status = HiveStatus(state: .offline, title: "Hive Rejected Request", subtitle: "Check the invite token.", peerCount: 0)
        } else {
            let detail = lastError.isEmpty ? "No saved LAN, tunnel, or relay endpoint answered." : lastError
            status = HiveStatus(state: .offline, title: "Hive Offline", subtitle: detail, peerCount: 0)
        }
    }

    private func probeNode(base: URL, token: String) async throws -> HiveProbeResult? {
        let probeURL = url(base: base, path: "api/hive/operator/status")
        var request = URLRequest(url: probeURL)
        request.timeoutInterval = 2
        request.setValue(token, forHTTPHeaderField: "X-Theseus-Hive-Secret")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { return nil }
        if http.statusCode == 401 || http.statusCode == 403 {
            throw HiveProbeError.rejected
        }
        guard (200..<300).contains(http.statusCode) else { return nil }
        return parseNodeStatus(data, base: base)
    }

    private func probeRelay(base: URL, hiveID: String, token: String) async throws -> HiveProbeResult? {
        guard !hiveID.isEmpty else { return nil }
        let probeURL = url(base: base, path: "api/hive/relay/peers", queryItems: [URLQueryItem(name: "hive_id", value: hiveID)])
        var request = URLRequest(url: probeURL)
        request.timeoutInterval = 5
        request.setValue(token, forHTTPHeaderField: "X-Theseus-Hive-Secret")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { return nil }
        if http.statusCode == 401 || http.statusCode == 403 {
            throw HiveProbeError.rejected
        }
        guard (200..<300).contains(http.statusCode) else { return nil }
        return parseRelayStatus(data, base: base)
    }

    private func parseNodeStatus(_ data: Data, base: URL) -> HiveProbeResult {
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let hive = object["hive"] as? [String: Any]
        else {
            return HiveProbeResult(status: HiveStatus(state: .offline, title: "Unreadable Status", subtitle: "Hive returned an unexpected response.", peerCount: 0), learnedEndpoints: [])
        }
        let local = hive["local_node"] as? [String: Any] ?? [:]
        let peerCount = hive["peer_count"] as? Int ?? 0
        let nodeName = local["node_name"] as? String ?? "Theseus Hive"
        let utilization = object["utilization"] as? [String: Any] ?? [:]
        let utilizationSummary = utilization["summary"] as? [String: Any] ?? [:]
        let accelerators = object["accelerators"] as? [String: Any] ?? [:]
        let appleMlx = accelerators["apple_mlx"] as? [String: Any] ?? [:]
        let utilizationState = utilization["trigger_state"] as? String ?? "--"
        let blockedNodes = utilizationSummary["blocked_nodes"] as? Int ?? utilization["blocked_nodes"] as? Int ?? 0
        let mlxReady = appleMlx["available"] as? Bool ?? false
        let mlxNodeCount = appleMlx["node_count"] as? Int ?? 0
        let mlxSummary = mlxReady ? "MLX \(max(mlxNodeCount, 1)) ready" : "MLX missing"
        let state: HiveStatus.State = peerCount > 0 ? .connected : .running
        let title = state == .connected ? "Connected" : "Running Locally"
        let subtitle = peerCount > 0 ? "\(nodeName) sees \(peerCount) peer(s)." : "\(nodeName) is online at \(base.host ?? "Hive")."
        return HiveProbeResult(
            status: HiveStatus(
                state: state,
                title: title,
                subtitle: subtitle,
                peerCount: peerCount,
                utilizationState: utilizationState,
                blockedNodeCount: blockedNodes,
                mlxSummary: mlxSummary,
                activeEndpoint: base.host ?? base.absoluteString
            ),
            learnedEndpoints: learnedEndpoints(from: object)
        )
    }

    private func parseRelayStatus(_ data: Data, base: URL) -> HiveProbeResult {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return HiveProbeResult(status: HiveStatus(state: .offline, title: "Unreadable Relay", subtitle: "Relay returned an unexpected response.", peerCount: 0), learnedEndpoints: [])
        }
        let peers = object["peers"] as? [[String: Any]] ?? []
        let peerCount = peers.count
        let state: HiveStatus.State = peerCount > 0 ? .connected : .running
        let title = peerCount > 0 ? "Relay Connected" : "Relay Online"
        let subtitle = peerCount > 0 ? "\(peerCount) node(s) reachable through \(base.host ?? "relay")." : "Relay is online; waiting for nodes to poll."
        return HiveProbeResult(
            status: HiveStatus(
                state: state,
                title: title,
                subtitle: subtitle,
                peerCount: peerCount,
                utilizationState: "relay",
                blockedNodeCount: 0,
                mlxSummary: "MLX via peers",
                activeEndpoint: base.host ?? base.absoluteString
            ),
            learnedEndpoints: learnedEndpoints(fromPeers: peers)
        )
    }

    private func learnedEndpoints(from object: [String: Any]) -> [String] {
        var raw: [String] = []
        if let roaming = object["roaming"] as? [String: Any] {
            appendStringArray(roaming["node_urls"], to: &raw)
            appendStringArray(roaming["relay_urls"], to: &raw)
            if let endpoints = roaming["endpoints"] as? [[String: Any]] {
                for endpoint in endpoints {
                    appendString(endpoint["url"], to: &raw)
                }
            }
        }
        if let hive = object["hive"] as? [String: Any] {
            if let local = hive["local_node"] as? [String: Any] {
                appendString(local["api_url"], to: &raw)
                appendString(local["relay_url"], to: &raw)
            }
            if let peers = hive["peers"] as? [[String: Any]] {
                raw += learnedEndpoints(fromPeers: peers)
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

    private func appendStringArray(_ value: Any?, to raw: inout [String]) {
        guard let rows = value as? [Any] else { return }
        for row in rows {
            if let text = row as? String {
                raw.append(text)
            } else if let endpoint = row as? [String: Any] {
                appendString(endpoint["url"], to: &raw)
            }
        }
    }

    private func url(base: URL, path: String, queryItems: [URLQueryItem] = []) -> URL {
        var components = URLComponents(url: base, resolvingAgainstBaseURL: false)
        components?.path = "/" + path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        components?.queryItems = queryItems.isEmpty ? nil : queryItems
        return components?.url ?? base.appendingPathComponent(path)
    }
}

private struct HiveProbeResult {
    let status: HiveStatus
    let learnedEndpoints: [String]
}

private enum HiveProbeError: Error {
    case rejected
}
