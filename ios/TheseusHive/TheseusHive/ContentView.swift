import SwiftUI

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @EnvironmentObject private var settings: HiveSettings
    @EnvironmentObject private var monitor: HiveMonitor
    @EnvironmentObject private var notifications: HiveNotificationRelay
    @State private var showingSettings = false
    @State private var reloadID = UUID()

    var body: some View {
        NavigationStack {
            ZStack {
                Color(.systemBackground).ignoresSafeArea()
                if settings.isConfigured, let url = settings.mobileURL {
                    VStack(spacing: 0) {
                        StatusStrip(status: monitor.status, unreadCount: notifications.unreadCount)
                        HiveWebView(url: url, token: settings.token, reloadID: reloadID)
                            .ignoresSafeArea(edges: .bottom)
                    }
                } else {
                    OnboardingView(showingSettings: $showingSettings)
                }
            }
            .navigationTitle("Theseus Hive")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        showingSettings = true
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                    }
                    .accessibilityLabel("Hive settings")
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        reloadID = UUID()
                        Task { await monitor.refresh(settings: settings) }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("Refresh Hive")
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsView()
                    .environmentObject(settings)
                    .environmentObject(monitor)
            }
            .task {
                await monitor.refresh(settings: settings)
                notifications.start(settings: settings)
                await notifications.pollOnce(settings: settings)
            }
            .onChange(of: settings.nodeURLString) { _ in
                reloadID = UUID()
                WatchProfileBridge.shared.send(settings: settings)
                notifications.start(settings: settings)
                Task { await monitor.refresh(settings: settings) }
                Task { await notifications.pollOnce(settings: settings) }
            }
            .onChange(of: settings.token) { _ in
                reloadID = UUID()
                WatchProfileBridge.shared.send(settings: settings)
                notifications.start(settings: settings)
                Task { await monitor.refresh(settings: settings) }
                Task { await notifications.pollOnce(settings: settings) }
            }
            .onChange(of: scenePhase) { phase in
                guard phase == .active else {
                    notifications.stop()
                    return
                }
                notifications.start(settings: settings)
                Task { await monitor.refresh(settings: settings) }
                Task { await notifications.pollOnce(settings: settings) }
            }
            .onOpenURL { url in
                do {
                    try settings.applyJoinURL(url)
                    reloadID = UUID()
                    Task { await monitor.refresh(settings: settings) }
                } catch {
                    // The settings sheet can still import the same invite JSON manually.
                }
            }
        }
    }
}

private struct StatusStrip: View {
    let status: HiveStatus
    let unreadCount: Int

    var body: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(status.color)
                .frame(width: 10, height: 10)
            VStack(alignment: .leading, spacing: 1) {
                Text(status.title)
                    .font(.subheadline.weight(.semibold))
                Text(status.subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                HStack(spacing: 6) {
                    StatusChip(status.utilizationState)
                    StatusChip(status.mlxSummary)
                    if unreadCount > 0 {
                        StatusChip("\(unreadCount) alert\(unreadCount == 1 ? "" : "s")", tint: .orange)
                    }
                    if status.blockedNodeCount > 0 {
                        StatusChip("\(status.blockedNodeCount) blocked", tint: .red)
                    }
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 1) {
                Text("\(status.peerCount)")
                    .font(.headline.monospacedDigit())
                Text("peers")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.regularMaterial)
        .overlay(alignment: .bottom) {
            Divider()
        }
    }
}

private struct StatusChip: View {
    let text: String
    let tint: Color

    init(_ text: String, tint: Color = .secondary) {
        self.text = text
        self.tint = tint
    }

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(tint)
            .lineLimit(1)
            .minimumScaleFactor(0.75)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Capsule().fill(Color(uiColor: .secondarySystemFill)))
    }
}

private struct OnboardingView: View {
    @Binding var showingSettings: Bool

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "hexagon.fill")
                .font(.system(size: 58))
                .foregroundStyle(.tint)
            VStack(spacing: 8) {
                Text("Connect to Theseus Hive")
                    .font(.title2.weight(.bold))
                Text("Add a trusted node URL and invite token to use the private operator from this iPhone.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            Button {
                showingSettings = true
            } label: {
                Label("Set Up Hive Access", systemImage: "key.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding(28)
    }
}
