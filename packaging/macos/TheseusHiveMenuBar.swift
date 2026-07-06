import AppKit
import Foundation

private struct HiveSnapshot {
    var state: State = .offline
    var nodeName: String = "Project Theseus Hive"
    var hiveId: String = "--"
    var version: String = "--"
    var peerCount: Int = 0
    var capabilities: [String] = []
    var mlxReady: Bool = false
    var mlxNodeCount: Int = 0
    var mlxTaskKinds: [String] = []
    var allowedTaskKinds: [String] = []
    var utilizationState: String = "--"
    var activeOrPlannedNodes: Int = 0
    var totalUtilizationNodes: Int = 0
    var plannedActions: Int = 0
    var executedActions: Int = 0
    var blockedNodes: Int = 0
    var detail: String = "Hive service is not responding."

    enum State {
        case connected
        case running
        case offline

        var title: String {
            switch self {
            case .connected: return "Connected"
            case .running: return "Running"
            case .offline: return "Offline"
            }
        }

        var glyph: String {
            switch self {
            case .connected: return "T"
            case .running: return "T"
            case .offline: return "T"
            }
        }

        var color: NSColor {
            switch self {
            case .connected: return .systemGreen
            case .running: return .systemYellow
            case .offline: return .systemRed
            }
        }
    }
}

@main
final class TheseusHiveMenuBarApp: NSObject, NSApplicationDelegate {
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    private let localBaseURL = URL(string: "http://127.0.0.1:8791")!
    private let dashboardURL = URL(string: "http://127.0.0.1:8787")!
    private var timer: Timer?
    private var snapshot = HiveSnapshot()

    private var supportRoot: URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Application Support/Project Theseus Hive", isDirectory: true)
    }

    private var installRoot: URL {
        supportRoot.appendingPathComponent("app/current", isDirectory: true)
    }

    private var runtimeRoot: URL {
        supportRoot.appendingPathComponent("runtime", isDirectory: true)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        configureStatusItem()
        refreshNow()
        timer = Timer.scheduledTimer(withTimeInterval: 8, repeats: true) { [weak self] _ in
            self?.refreshNow()
        }
    }

    private func configureStatusItem() {
        statusItem.button?.target = self
        statusItem.button?.action = #selector(openMenu)
        updateStatusButton()
        statusItem.menu = buildMenu()
    }

    @objc private func openMenu() {
        statusItem.menu = buildMenu()
        statusItem.button?.performClick(nil)
    }

    private func updateStatusButton() {
        let attributes: [NSAttributedString.Key: Any] = [
            .foregroundColor: snapshot.state.color,
            .font: NSFont.systemFont(ofSize: 14, weight: .bold)
        ]
        statusItem.button?.attributedTitle = NSAttributedString(string: " \(snapshot.state.glyph) ", attributes: attributes)
        statusItem.button?.toolTip = "Theseus Hive: \(snapshot.state.title)"
    }

    private func buildMenu() -> NSMenu {
        let menu = NSMenu()
        menu.addItem(disabledItem("Theseus Hive \(snapshot.state.title)"))
        menu.addItem(disabledItem(snapshot.nodeName))
        menu.addItem(disabledItem("Hive: \(snapshot.hiveId)"))
        menu.addItem(disabledItem("Peers: \(snapshot.peerCount) / Version: \(snapshot.version)"))
        menu.addItem(disabledItem("Apple MLX: \(snapshot.mlxReady ? "Ready" : "Missing") (\(snapshot.mlxNodeCount) node\(snapshot.mlxNodeCount == 1 ? "" : "s"))"))
        menu.addItem(disabledItem("Always Active: \(snapshot.utilizationState) / \(snapshot.activeOrPlannedNodes)/\(max(snapshot.totalUtilizationNodes, 1)) covered / \(snapshot.blockedNodes) blocked"))
        menu.addItem(disabledItem("Queued: \(snapshot.plannedActions) planned / \(snapshot.executedActions) executed"))
        if !snapshot.capabilities.isEmpty {
            menu.addItem(disabledItem(snapshot.capabilities.prefix(5).joined(separator: ", ")))
        }
        menu.addItem(NSMenuItem.separator())
        menu.addItem(actionItem("Open Operator / Chat", action: #selector(openOperator)))
        menu.addItem(actionItem("Open Dashboard", action: #selector(openDashboard)))
        menu.addItem(actionItem("Open Logs", action: #selector(openLogs)))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(actionItem("Run Always-Active Sweep", action: #selector(queueUtilizationSweep), enabled: snapshot.allowedTaskKinds.contains("utilization_sweep")))
        menu.addItem(actionItem("Queue Training Round", action: #selector(queueTrainingRound), enabled: snapshot.allowedTaskKinds.contains("training_orchestrate")))
        menu.addItem(actionItem("Pause Always-Active Loop", action: #selector(pauseUtilization), enabled: snapshot.allowedTaskKinds.contains("utilization_sweep")))
        menu.addItem(actionItem("Resume Always-Active Loop", action: #selector(resumeUtilization), enabled: snapshot.allowedTaskKinds.contains("utilization_sweep")))
        menu.addItem(actionItem("Stop Always-Active Loop", action: #selector(stopUtilization), enabled: snapshot.allowedTaskKinds.contains("utilization_sweep")))
        menu.addItem(actionItem("Clear Stop Flag", action: #selector(clearUtilizationStop), enabled: snapshot.allowedTaskKinds.contains("utilization_sweep")))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(actionItem("Queue MLX Eval Smoke", action: #selector(queueMlxEval), enabled: snapshot.mlxTaskKinds.contains("mlx_eval_chunk")))
        menu.addItem(actionItem("Queue MLX Train Smoke", action: #selector(queueMlxTrain), enabled: snapshot.mlxTaskKinds.contains("mlx_training_chunk")))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(actionItem("Refresh Status", action: #selector(refreshNow)))
        menu.addItem(actionItem("Restart Hive Service", action: #selector(restartHive)))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(actionItem("Quit Menu Bar Icon", action: #selector(quit)))
        return menu
    }

    private func disabledItem(_ title: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.isEnabled = false
        return item
    }

    private func actionItem(_ title: String, action: Selector, enabled: Bool = true) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
        item.target = self
        item.isEnabled = enabled
        return item
    }

    @objc private func refreshNow() {
        var request = URLRequest(url: localBaseURL.appendingPathComponent("api/hive/operator/status"))
        request.timeoutInterval = 4
        URLSession.shared.dataTask(with: request) { [weak self] data, _, error in
            let next = self?.parseSnapshot(data: data, error: error) ?? HiveSnapshot()
            DispatchQueue.main.async {
                self?.snapshot = next
                self?.updateStatusButton()
                self?.statusItem.menu = self?.buildMenu()
            }
        }.resume()
    }

    private func parseSnapshot(data: Data?, error: Error?) -> HiveSnapshot {
        guard error == nil, let data else {
            return HiveSnapshot(detail: error?.localizedDescription ?? "Hive service is not responding.")
        }
        guard
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let hive = root["hive"] as? [String: Any]
        else {
            return HiveSnapshot(detail: "Hive returned an unreadable status payload.")
        }

        let local = hive["local_node"] as? [String: Any] ?? [:]
        let updates = local["updates"] as? [String: Any] ?? [:]
        let hiveVersion = local["hive_version"] as? [String: Any] ?? [:]
        let peerCount = hive["peer_count"] as? Int ?? 0
        let targets = hive["targets"] as? [[String: Any]] ?? []
        let accelerators = root["accelerators"] as? [String: Any] ?? [:]
        let appleMlx = accelerators["apple_mlx"] as? [String: Any] ?? [:]
        let capabilities = (local["capabilities"] as? [[String: Any]] ?? [])
            .compactMap { $0["id"] as? String }
        let utilization = root["utilization"] as? [String: Any] ?? [:]
        let utilizationSummary = utilization["summary"] as? [String: Any] ?? [:]
        let utilizationNodes = utilization["nodes"] as? [[String: Any]] ?? []
        let allowedTaskKinds = hive["allowed_task_kinds"] as? [String] ?? []

        let version = (hiveVersion["local_version_id"] as? String)
            ?? (updates["version_id"] as? String)
            ?? "--"
        let state: HiveSnapshot.State = peerCount > 0 || targets.count > 1 ? .connected : .running
        return HiveSnapshot(
            state: state,
            nodeName: local["node_name"] as? String ?? "Project Theseus Hive",
            hiveId: hive["hive_id"] as? String ?? "--",
            version: version,
            peerCount: peerCount,
            capabilities: capabilities,
            mlxReady: appleMlx["available"] as? Bool ?? false,
            mlxNodeCount: appleMlx["node_count"] as? Int ?? 0,
            mlxTaskKinds: appleMlx["task_kinds"] as? [String] ?? [],
            allowedTaskKinds: allowedTaskKinds,
            utilizationState: utilization["trigger_state"] as? String ?? "--",
            activeOrPlannedNodes: intValue(utilizationSummary["active_or_planned_nodes"] ?? utilization["active_or_planned_nodes"]),
            totalUtilizationNodes: utilizationNodes.count,
            plannedActions: intValue(utilizationSummary["planned_actions"] ?? utilization["planned_actions"]),
            executedActions: intValue(utilizationSummary["executed_actions"] ?? utilization["executed_actions"]),
            blockedNodes: intValue(utilizationSummary["blocked_nodes"] ?? utilization["blocked_nodes"]),
            detail: "OK"
        )
    }

    @objc private func openOperator() {
        NSWorkspace.shared.open(localBaseURL.appendingPathComponent("mobile"))
    }

    @objc private func openDashboard() {
        NSWorkspace.shared.open(dashboardURL)
    }

    @objc private func openLogs() {
        NSWorkspace.shared.open(runtimeRoot.appendingPathComponent("logs", isDirectory: true))
    }

    @objc private func queueMlxEval() {
        submitTask(kind: "mlx_eval_chunk", payload: [
            "profile": "smoke",
            "chunk_id": "menubar_mlx_eval",
            "steps": 1,
            "eval_limit": 128,
            "train_limit": 128
        ])
    }

    @objc private func queueMlxTrain() {
        submitTask(kind: "mlx_training_chunk", payload: [
            "profile": "smoke",
            "chunk_id": "menubar_mlx_train",
            "steps": 4,
            "eval_limit": 128,
            "train_limit": 256
        ])
    }

    @objc private func queueTrainingRound() {
        submitTask(kind: "training_orchestrate", payload: [
            "profile": "smoke",
            "source": "menubar",
            "sync_artifacts": true
        ])
    }

    @objc private func queueUtilizationSweep() {
        utilizationControl(action: "sweep")
    }

    @objc private func pauseUtilization() {
        utilizationControl(action: "pause")
    }

    @objc private func resumeUtilization() {
        utilizationControl(action: "resume")
    }

    @objc private func stopUtilization() {
        utilizationControl(action: "stop")
    }

    @objc private func clearUtilizationStop() {
        utilizationControl(action: "clear_stop")
    }

    private func submitTask(kind: String, payload: [String: Any]) {
        var request = URLRequest(url: localBaseURL.appendingPathComponent("api/hive/operator/task"))
        request.httpMethod = "POST"
        request.timeoutInterval = 8
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: [
            "target_node_id": "local",
            "kind": kind,
            "task_payload": payload
        ])
        URLSession.shared.dataTask(with: request) { [weak self] _, _, _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                self?.refreshNow()
            }
        }.resume()
    }

    private func utilizationControl(action: String) {
        var request = URLRequest(url: localBaseURL.appendingPathComponent("api/hive/operator/utilization"))
        request.httpMethod = "POST"
        request.timeoutInterval = 8
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["action": action])
        URLSession.shared.dataTask(with: request) { [weak self] _, _, _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                self?.refreshNow()
            }
        }.resume()
    }

    @objc private func restartHive() {
        run("/bin/launchctl", ["kickstart", "-k", "gui/\(getuid())/local.project-theseus.hive"])
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            self?.refreshNow()
        }
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }

    private func run(_ executable: String, _ arguments: [String]) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        try? process.run()
    }

    private func intValue(_ value: Any?) -> Int {
        if let int = value as? Int {
            return int
        }
        if let number = value as? NSNumber {
            return number.intValue
        }
        if let text = value as? String, let int = Int(text) {
            return int
        }
        return 0
    }
}
