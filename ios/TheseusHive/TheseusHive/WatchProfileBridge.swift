import Foundation
import WatchConnectivity

@MainActor
final class WatchProfileBridge: NSObject {
    static let shared = WatchProfileBridge()

    private var lastPayload: [String: Any] = [:]

    private override init() {
        super.init()
    }

    func start(settings: HiveSettings) {
        guard WCSession.isSupported() else { return }
        let session = WCSession.default
        if session.delegate == nil {
            session.delegate = self
            session.activate()
        }
        send(settings: settings)
    }

    func send(settings: HiveSettings) {
        guard WCSession.isSupported() else { return }
        let payload: [String: Any] = [
            "policy": "project_theseus_watch_profile_v0",
            "node_url": settings.nodeURLString,
            "endpoint_urls": settings.endpointURLStrings,
            "hive_id": settings.hiveID,
            "operator_token": settings.token,
            "updated_utc": ISO8601DateFormatter().string(from: Date())
        ]
        lastPayload = payload
        let session = WCSession.default
        guard session.activationState == .activated else { return }
        do {
            try session.updateApplicationContext(payload)
        } catch {
            if session.isReachable {
                session.sendMessage(payload, replyHandler: nil, errorHandler: nil)
            } else {
                session.transferUserInfo(payload)
            }
        }
    }

    func sendNotification(_ notification: HiveOperatorNotification) {
        guard WCSession.isSupported() else { return }
        let payload: [String: Any] = [
            "policy": "project_theseus_watch_notification_v0",
            "id": notification.id,
            "severity": notification.severity,
            "title": notification.title,
            "body": notification.body,
            "category": notification.category,
            "created_utc": notification.createdUTC
        ]
        let session = WCSession.default
        guard session.activationState == .activated else { return }
        if session.isReachable {
            session.sendMessage(payload, replyHandler: nil, errorHandler: nil)
        }
        session.transferUserInfo(payload)
    }
}

extension WatchProfileBridge: WCSessionDelegate {
    nonisolated func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {
        guard activationState == .activated else { return }
        Task { @MainActor in
            if !lastPayload.isEmpty {
                try? session.updateApplicationContext(lastPayload)
            }
        }
    }

    nonisolated func sessionDidBecomeInactive(_ session: WCSession) {}

    nonisolated func sessionDidDeactivate(_ session: WCSession) {
        session.activate()
    }
}
