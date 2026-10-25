import SwiftUI
import WorkbenchCore

/// The tutor side panel: a coaching chat that sees the open lesson, the
/// learner's file, the last failure, their prediction and their mistakes.
struct TutorPanel: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openSettings) private var openSettings
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if !model.hasAPIKey {
                ContentUnavailableView {
                    Label("Add an API key", systemImage: "key")
                } description: {
                    Text("The tutor runs on the Claude API. Add your key in Settings; it's stored in your Keychain.")
                } actions: {
                    Button("Open Settings") { openSettings() }
                }
            } else if model.tutor.messages.isEmpty {
                ContentUnavailableView {
                    Label("Ask the tutor", systemImage: "graduationcap")
                } description: {
                    Text("It sees your lesson, your file, your last error and your prediction, and it coaches rather than hands out answers. Use \"Explain\" on an instrument to ask about a tensor.")
                }
            } else {
                transcript
            }
            Divider()
            input
        }
        .frame(minWidth: 280)
    }

    private var header: some View {
        HStack {
            Text("Tutor").font(.headline)
            Spacer()
            Button("Clear") { model.clearTutor() }
                .buttonStyle(.borderless)
                .disabled(model.tutor.messages.isEmpty)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(model.tutor.messages) { message in
                        TutorMessageView(message: message, streaming: model.tutor.isAnswering
                                         && message.id == model.tutor.messages.last?.id)
                            .id(message.id)
                    }
                }
                .padding(12)
            }
            .onChange(of: model.tutor.messages.last?.text) { _, _ in
                if let last = model.tutor.messages.last?.id {
                    proxy.scrollTo(last, anchor: .bottom)
                }
            }
        }
    }

    private var input: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Ask about this lesson…", text: $model.tutorDraft, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...6)
                .focused($inputFocused)
                .onSubmit { model.askTutor() }
                .disabled(!model.hasAPIKey)
            if model.tutor.isAnswering {
                Button { model.stopTutor() } label: { Image(systemName: "stop.circle.fill") }
                    .help("Stop the answer")
            } else {
                Button { model.askTutor() } label: { Image(systemName: "arrow.up.circle.fill") }
                    .disabled(model.tutorDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                              || !model.hasAPIKey)
                    .keyboardShortcut(.return, modifiers: .command)
                    .help("Send (Return)")
            }
        }
        .buttonStyle(.borderless)
        .font(.body)
        .padding(10)
        .onAppear { inputFocused = true }
    }
}

private struct TutorMessageView: View {
    let message: TutorChat.Message
    let streaming: Bool

    var body: some View {
        VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
            if message.role == .user {
                Text(message.text)
                    .textSelection(.enabled)
                    .padding(8)
                    .background(Color.accentColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 8))
                    .frame(maxWidth: .infinity, alignment: .trailing)
            } else if message.isError {
                Label(message.text, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
            } else if message.text.isEmpty && streaming {
                ProgressView().controlSize(.small)
            } else {
                MarkdownView(source: message.text)
            }
            if let note = message.note {
                Text(note).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

/// Settings window (⌘,): the tutor's API key.
struct SettingsView: View {
    @EnvironmentObject private var model: AppModel
    @State private var key = ""

    var body: some View {
        Form {
            Section {
                SecureField("Claude API key", text: $key, prompt: Text("sk-ant-…"))
                HStack {
                    Button("Save to Keychain") {
                        model.saveAPIKey(key)
                        key = ""
                    }
                    .disabled(key.trimmingCharacters(in: .whitespaces).isEmpty)
                    Button("Remove") { model.saveAPIKey("") }
                    Spacer()
                    Text(model.hasAPIKey ? "A key is set." : "No key yet.")
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Tutor")
            } footer: {
                Text("Get a key at console.anthropic.com. It's stored in your login Keychain and sent only to the Claude API. ANTHROPIC_API_KEY in the environment also works.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 460)
        .padding()
    }
}
