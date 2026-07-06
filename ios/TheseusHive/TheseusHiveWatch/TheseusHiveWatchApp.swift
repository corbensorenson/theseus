import SwiftUI

@main
struct TheseusHiveWatchApp: App {
    @StateObject private var settings = WatchHiveSettings()
    @StateObject private var monitor = WatchHiveMonitor()
    @StateObject private var bridge = WatchProfileReceiver()

    var body: some Scene {
        WindowGroup {
            WatchContentView()
                .environmentObject(settings)
                .environmentObject(monitor)
                .environmentObject(bridge)
                .onAppear {
                    bridge.start(settings: settings)
                    if settings.notificationsEnabled && settings.isConfigured {
                        Task { await WatchHiveNotifier.requestAuthorization() }
                    }
                }
                .onProfileUpdate(of: bridge.lastProfileUTC) {
                    Task { await monitor.refresh(settings: settings) }
                }
        }
    }
}

private extension View {
    @ViewBuilder
    func onProfileUpdate<Value: Equatable>(of value: Value, perform action: @escaping () -> Void) -> some View {
        if #available(watchOS 10.0, *) {
            onChange(of: value) {
                action()
            }
        } else {
            onChange(of: value) { _ in
                action()
            }
        }
    }
}
