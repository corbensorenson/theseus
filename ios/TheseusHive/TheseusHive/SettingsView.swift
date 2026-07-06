import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var settings: HiveSettings
    @EnvironmentObject private var monitor: HiveMonitor

    @State private var nodeURL = ""
    @State private var endpointURLs = ""
    @State private var hiveID = ""
    @State private var token = ""
    @State private var inviteJSON = ""
    @State private var message = ""
    @State private var showingScanner = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Connection") {
                    TextField("Best or last-good URL", text: $nodeURL)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                    TextField("Hive ID", text: $hiveID)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField("Hive invite token", text: $token)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }

                Section("Roaming URLs") {
                    Text("The app tries these in order, so it can move between home Wi-Fi, hotspot, private tunnel, and relay without changing settings.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    TextEditor(text: $endpointURLs)
                        .frame(minHeight: 110)
                        .font(.system(.footnote, design: .monospaced))
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }

                Section("Notifications") {
                    Toggle(
                        "Hive Alerts",
                        isOn: Binding(
                            get: { settings.notificationsEnabled },
                            set: { settings.setNotificationsEnabled($0) }
                        )
                    )
                    Text("When this app is active, it checks the Hive notification feed and relays important alerts to the paired Apple Watch.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Import Invite") {
                    TextEditor(text: $inviteJSON)
                        .frame(minHeight: 140)
                        .font(.system(.footnote, design: .monospaced))
                    Button {
                        importInvite()
                    } label: {
                        Label("Import Invite JSON", systemImage: "doc.text.magnifyingglass")
                    }
                    Button {
                        showingScanner = true
                    } label: {
                        Label("Scan Join QR", systemImage: "qrcode.viewfinder")
                    }
                }

                Section {
                    Button {
                        save()
                    } label: {
                        Label("Save And Connect", systemImage: "checkmark.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)

                    Button(role: .destructive) {
                        settings.clear()
                        nodeURL = ""
                        token = ""
                        message = "Local connection settings cleared."
                    } label: {
                        Label("Clear Local Settings", systemImage: "trash")
                    }
                }

                if !message.isEmpty {
                    Section("Status") {
                        Text(message)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Hive Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .onAppear {
                syncFieldsFromSettings()
            }
            .sheet(isPresented: $showingScanner) {
                QRScannerView { value in
                    showingScanner = false
                    importScannedProfile(value)
                } onError: { errorMessage in
                    showingScanner = false
                    message = errorMessage
                }
                .ignoresSafeArea()
            }
        }
    }

    private func importInvite() {
        do {
            let invite = try settings.parseInvite(inviteJSON)
            try apply(invite: invite)
        } catch {
            message = error.localizedDescription
        }
    }

    private func importScannedProfile(_ raw: String) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            if let url = URL(string: trimmed), url.scheme == "theseushive" {
                try settings.applyJoinURL(url)
                syncFieldsFromSettings()
                WatchProfileBridge.shared.send(settings: settings)
                message = "Hive profile imported. Checking status..."
                Task { await refreshStatusMessage() }
                return
            }
            let invite = try settings.parseInvite(trimmed)
            try apply(invite: invite)
        } catch {
            message = error.localizedDescription
        }
    }

    private func apply(invite: HiveInvite) throws {
        try settings.save(
            nodeURLString: invite.coordinatorURL,
            endpointURLStrings: invite.endpointURLs,
            hiveID: invite.hiveID,
            token: invite.joinToken
        )
        syncFieldsFromSettings()
        WatchProfileBridge.shared.send(settings: settings)
        message = "Hive profile imported with \(invite.endpointURLs.count) endpoint(s). Checking status..."
        Task { await refreshStatusMessage() }
    }

    private func save() {
        do {
            let endpoints = endpointURLs
                .split(whereSeparator: \.isNewline)
                .map { String($0) }
            try settings.save(nodeURLString: nodeURL, endpointURLStrings: endpoints, hiveID: hiveID, token: token)
            WatchProfileBridge.shared.send(settings: settings)
            message = "Saved. Checking Hive status..."
            Task {
                await refreshStatusMessage()
            }
        } catch {
            message = error.localizedDescription
        }
    }

    private func syncFieldsFromSettings() {
        nodeURL = settings.nodeURLString
        endpointURLs = settings.endpointURLText
        hiveID = settings.hiveID
        token = settings.token
    }

    private func refreshStatusMessage() async {
        await monitor.refresh(settings: settings)
        await MainActor.run {
            message = monitor.status.title
        }
    }
}
