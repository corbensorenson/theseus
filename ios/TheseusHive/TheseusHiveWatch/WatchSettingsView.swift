import SwiftUI

struct WatchSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var settings: WatchHiveSettings
    @EnvironmentObject private var monitor: WatchHiveMonitor

    @State private var nodeURL = ""
    @State private var hiveID = ""
    @State private var token = ""
    @State private var message = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Profile") {
                    TextField("Node URL", text: $nodeURL)
                        .textInputAutocapitalization(.never)
                    TextField("Hive ID", text: $hiveID)
                        .textInputAutocapitalization(.never)
                    SecureField("Token", text: $token)
                        .textInputAutocapitalization(.never)
                }
                Section("Watch") {
                    Toggle(
                        "Notifications",
                        isOn: Binding(
                            get: { settings.notificationsEnabled },
                            set: { settings.setNotificationsEnabled($0) }
                        )
                    )
                }
                Section {
                    Button {
                        settings.save(nodeURLString: nodeURL, endpointURLStrings: [nodeURL], hiveID: hiveID, token: token)
                        message = "Saved."
                        Task { await monitor.refresh(settings: settings) }
                    } label: {
                        Label("Save", systemImage: "checkmark")
                    }
                    Button(role: .destructive) {
                        settings.clear()
                        nodeURL = ""
                        hiveID = ""
                        token = ""
                        message = "Cleared."
                    } label: {
                        Label("Clear", systemImage: "trash")
                    }
                }
                if !message.isEmpty {
                    Section {
                        Text(message)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Hive")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .onAppear {
                nodeURL = settings.nodeURLString
                hiveID = settings.hiveID
                token = settings.token
            }
        }
    }
}
