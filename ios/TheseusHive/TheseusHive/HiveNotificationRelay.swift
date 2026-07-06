import Foundation
import UserNotifications

struct HiveOperatorNotification: Identifiable {
    let id: String
    let severity: String
    let title: String
    let body: String
    let category: String
    let createdUTC: String
    let interruptive: Bool
    let acknowledged: Bool
}

@MainActor
final class HiveNotificationRelay: ObservableObject {
    @Published private(set) var unreadCount = 0
    @Published private(set) var latest: [HiveOperatorNotification] = []
    @Published private(set) var lastCheckedUTC = ""

    private var pollTask: Task<Void, Never>?
    private let seenKey = "theseus.hive.seenNotificationIDs"
    private let pollIntervalNanoseconds: UInt64 = 60 * 1_000_000_000

    func start(settings: HiveSettings) {
        guard pollTask == nil else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.pollOnce(settings: settings)
                try? await Task.sleep(nanoseconds: self?.pollIntervalNanoseconds ?? 60 * 1_000_000_000)
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    func pollOnce(settings: HiveSettings) async {
        guard settings.notificationsEnabled, settings.isConfigured else {
            unreadCount = 0
            latest = []
            return
        }
        for base in settings.endpointBaseURLs {
            do {
                let result = try await fetchNotifications(base: base, token: settings.token)
                settings.markActiveEndpoint(base)
                unreadCount = result.unreadCount
                latest = result.notifications
                lastCheckedUTC = ISO8601DateFormatter().string(from: Date())
                await deliverNewNotifications(result.notifications, settings: settings, base: base)
                return
            } catch {
                continue
            }
        }
    }

    private func fetchNotifications(base: URL, token: String) async throws -> HiveNotificationFetch {
        let url = endpoint(base: base, path: "api/hive/operator/notifications", queryItems: [URLQueryItem(name: "limit", value: "20")])
        var request = URLRequest(url: url)
        request.timeoutInterval = 8
        request.setValue(token, forHTTPHeaderField: "X-Theseus-Hive-Secret")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw HiveNotificationError.unreachable
        }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw HiveNotificationError.unreadable
        }
        let rows = object["notifications"] as? [[String: Any]] ?? []
        let notifications = rows.compactMap(parseNotification)
        let unread = object["unread_count"] as? Int ?? notifications.filter { !$0.acknowledged }.count
        return HiveNotificationFetch(unreadCount: unread, notifications: notifications)
    }

    private func parseNotification(_ row: [String: Any]) -> HiveOperatorNotification? {
        guard let id = row["id"] as? String, !id.isEmpty else { return nil }
        return HiveOperatorNotification(
            id: id,
            severity: row["severity"] as? String ?? "info",
            title: row["title"] as? String ?? "Theseus Hive",
            body: row["body"] as? String ?? "",
            category: row["category"] as? String ?? "hive",
            createdUTC: row["created_utc"] as? String ?? "",
            interruptive: row["interruptive"] as? Bool ?? false,
            acknowledged: row["acknowledged"] as? Bool ?? false
        )
    }

    private func deliverNewNotifications(_ notifications: [HiveOperatorNotification], settings: HiveSettings, base: URL) async {
        var seen = seenIDs()
        var delivered: [HiveOperatorNotification] = []
        for notification in notifications {
            guard notification.interruptive, !notification.acknowledged, !seen.contains(notification.id) else {
                continue
            }
            guard await requestAuthorizationIfNeeded() else { return }
            await deliverLocal(notification)
            WatchProfileBridge.shared.sendNotification(notification)
            seen.insert(notification.id)
            delivered.append(notification)
        }
        if !delivered.isEmpty {
            saveSeenIDs(seen)
            try? await acknowledge(ids: delivered.map(\.id), base: base, token: settings.token)
        }
    }

    private func requestAuthorizationIfNeeded() async -> Bool {
        let center = UNUserNotificationCenter.current()
        let state = await center.notificationSettings()
        switch state.authorizationStatus {
        case .authorized, .provisional:
            return true
        case .notDetermined:
            return (try? await center.requestAuthorization(options: [.alert, .sound, .badge])) ?? false
        default:
            return false
        }
    }

    private func deliverLocal(_ notification: HiveOperatorNotification) async {
        let content = UNMutableNotificationContent()
        content.title = notification.title
        content.body = notification.body
        content.sound = .default
        content.categoryIdentifier = "THESEUS_HIVE"
        let request = UNNotificationRequest(identifier: "theseus-hive-\(notification.id)", content: content, trigger: nil)
        try? await UNUserNotificationCenter.current().add(request)
    }

    private func acknowledge(ids: [String], base: URL, token: String) async throws {
        guard !ids.isEmpty else { return }
        let url = endpoint(base: base, path: "api/hive/operator/notifications/ack")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 8
        request.setValue(token, forHTTPHeaderField: "X-Theseus-Hive-Secret")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["ids": ids])
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw HiveNotificationError.unreachable
        }
    }

    private func endpoint(base: URL, path: String, queryItems: [URLQueryItem] = []) -> URL {
        var components = URLComponents(url: base, resolvingAgainstBaseURL: false)
        components?.path = "/" + path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        components?.queryItems = queryItems.isEmpty ? nil : queryItems
        return components?.url ?? base.appendingPathComponent(path)
    }

    private func seenIDs() -> Set<String> {
        Set(UserDefaults.standard.stringArray(forKey: seenKey) ?? [])
    }

    private func saveSeenIDs(_ ids: Set<String>) {
        let kept = Array(ids).suffix(300)
        UserDefaults.standard.set(Array(kept), forKey: seenKey)
    }
}

private struct HiveNotificationFetch {
    let unreadCount: Int
    let notifications: [HiveOperatorNotification]
}

private enum HiveNotificationError: Error {
    case unreadable
    case unreachable
}
