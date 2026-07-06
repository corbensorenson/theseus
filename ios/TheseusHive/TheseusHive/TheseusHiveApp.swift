import SwiftUI

@main
struct TheseusHiveApp: App {
    @StateObject private var settings = HiveSettings()
    @StateObject private var monitor = HiveMonitor()
    @StateObject private var notifications = HiveNotificationRelay()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
                .environmentObject(monitor)
                .environmentObject(notifications)
                .onAppear {
                    WatchProfileBridge.shared.start(settings: settings)
                }
        }
    }
}
