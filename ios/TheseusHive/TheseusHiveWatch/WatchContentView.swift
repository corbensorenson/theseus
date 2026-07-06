import SwiftUI

struct WatchContentView: View {
    @EnvironmentObject private var settings: WatchHiveSettings
    @EnvironmentObject private var monitor: WatchHiveMonitor
    @State private var commandText = ""
    @State private var showingSettings = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    StatusCard(status: monitor.status, busy: monitor.isBusy)
                    if settings.isConfigured {
                        CommandCard(commandText: $commandText)
                        QuickActions(commandText: $commandText)
                        if !monitor.lastMessage.isEmpty {
                            Text(monitor.lastMessage)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        SetupCard()
                    }
                }
                .padding(.horizontal, 2)
            }
            .navigationTitle("Theseus")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button {
                        showingSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                    .accessibilityLabel("Hive settings")
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button {
                        Task { await monitor.refresh(settings: settings) }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .accessibilityLabel("Refresh Hive")
                }
            }
            .sheet(isPresented: $showingSettings) {
                WatchSettingsView()
                    .environmentObject(settings)
                    .environmentObject(monitor)
            }
            .task {
                await monitor.refresh(settings: settings)
            }
        }
    }
}

private struct StatusCard: View {
    let status: WatchHiveStatus
    let busy: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 7) {
                Circle()
                    .fill(status.color)
                    .frame(width: 9, height: 9)
                Text(status.title)
                    .font(.headline)
                    .lineLimit(1)
                Spacer()
                if busy {
                    ProgressView()
                        .controlSize(.mini)
                }
            }
            Text(status.detail)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            HStack(spacing: 6) {
                StatPill(value: "\(status.peerCount)", label: "peers")
                StatPill(value: status.utilization, label: "work")
                StatPill(value: status.mlx, label: "accel")
            }
            HStack(spacing: 4) {
                Image(systemName: "point.3.connected.trianglepath.dotted")
                Text(status.activeEndpoint.isEmpty ? "No endpoint" : status.activeEndpoint)
                    .lineLimit(1)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            HStack(spacing: 4) {
                Image(systemName: "number")
                Text(status.version)
                    .lineLimit(1)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.14)))
    }
}

private struct StatPill: View {
    let value: String
    let label: String

    var body: some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.caption.weight(.bold))
                .lineLimit(1)
                .minimumScaleFactor(0.65)
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 30)
        .padding(.vertical, 3)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.12)))
    }
}

private struct CommandCard: View {
    @Binding var commandText: String
    @EnvironmentObject private var settings: WatchHiveSettings
    @EnvironmentObject private var monitor: WatchHiveMonitor

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            TextField("Dictate or type", text: $commandText)
                .textInputAutocapitalization(.sentences)
            Button {
                let text = commandText
                commandText = ""
                Task { await monitor.sendChat(text, settings: settings) }
            } label: {
                Label("Send", systemImage: "paperplane.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(commandText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.14)))
    }
}

private struct QuickActions: View {
    @Binding var commandText: String
    @EnvironmentObject private var settings: WatchHiveSettings
    @EnvironmentObject private var monitor: WatchHiveMonitor

    var body: some View {
        VStack(spacing: 7) {
            Button {
                Task { await monitor.checkpoint(settings: settings) }
            } label: {
                Label("Checkpoint", systemImage: "checkmark.seal")
                    .frame(maxWidth: .infinity)
            }
            Button {
                commandText = "What happened recently across the hive?"
                Task { await monitor.sendChat(commandText, settings: settings) }
                commandText = ""
            } label: {
                Label("Overnight", systemImage: "moon.stars")
                    .frame(maxWidth: .infinity)
            }
        }
        .buttonStyle(.bordered)
    }
}

private struct SetupCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: "applewatch")
                .font(.title2)
            Text("Open Theseus Hive on iPhone")
                .font(.headline)
            Text("The iPhone app sends the private roaming profile to this Watch. You can also enter a node URL and token in settings.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.14)))
    }
}
