import SwiftUI
import WorkbenchCore

/// The tutor side panel: a coaching chat that sees the open lesson, the
/// learner's file, the last failure, their prediction and their mistakes.
struct TutorPanel: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openSettings) private var openSettings
    @FocusState private var inputFocused: Bool

    private static let suggestions = [
        "Give me a hint, not the answer",
        "Why is my check failing?",
        "Explain the idea behind this lesson",
    ]

    var body: some View {
        VStack(spacing: 0) {
            header
            if !model.hasAPIKey {
                ContentUnavailableView {
                    Label("Add an API key", systemImage: "key.fill")
                } description: {
                    Text("The tutor runs on the Claude API. Add your key in Settings; it's stored in your Keychain.")
                } actions: {
                    Button("Open Settings") { openSettings() }
                        .prominentButton()
                }
                .frame(maxHeight: .infinity)
            } else if model.tutor.messages.isEmpty {
                emptyState
                    .frame(maxHeight: .infinity)
            } else {
                transcript
            }
            input
        }
        .frame(minWidth: 300)
        .background(Theme.Palette.background)
        .animation(Theme.Motion.smooth, value: model.tutor.messages.count)
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "asterisk")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Theme.accent)
            VStack(alignment: .leading, spacing: 0) {
                Text("Tutor").font(.system(size: 13, weight: .semibold))
                Text(model.tutor.isAnswering ? "Thinking…" : "Coaches, doesn't hand out answers")
                    .font(.caption)
                    .foregroundStyle(Theme.Palette.secondaryText)
                    .contentTransition(.opacity)
            }
            Spacer()
            Button {
                withAnimation(Theme.Motion.smooth) { model.clearTutor() }
            } label: {
                Image(systemName: "square.and.pencil")
            }
            .buttonStyle(.borderless)
            .help("New conversation")
            .disabled(model.tutor.messages.isEmpty)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private var emptyState: some View {
        VStack(spacing: Theme.Space.m) {
            Image(systemName: "asterisk")
                .font(.system(size: 26, weight: .semibold))
                .foregroundStyle(Theme.accent)
                .symbolEffect(.pulse, options: .repeating.speed(0.4))
            Text("Ask about this lesson")
                .font(Theme.display(22))
            Text("The tutor sees your lesson, your file, your last error and your prediction. Use Explain on an instrument to ask about a tensor.")
                .font(.callout)
                .foregroundStyle(Theme.Palette.secondaryText)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            VStack(spacing: 6) {
                ForEach(Self.suggestions, id: \.self) { suggestion in
                    Button {
                        model.askTutor(suggestion)
                    } label: {
                        Text(suggestion)
                            .frame(maxWidth: .infinity)
                    }
                    .secondaryButton()
                }
            }
            .padding(.top, 4)
        }
        .padding(Theme.Space.xl)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(model.tutor.messages) { message in
                        TutorMessageView(message: message, streaming: model.tutor.isAnswering
                                         && message.id == model.tutor.messages.last?.id)
                            .id(message.id)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
                .padding(14)
            }
            .thinScrollBar()
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
                .padding(.vertical, 4)
            if model.tutor.isAnswering {
                Button { model.stopTutor() } label: {
                    Image(systemName: "stop.circle.fill")
                        .font(.system(size: 22))
                        .foregroundStyle(Theme.Palette.secondaryText)
                }
                .help("Stop the answer")
                .transition(.scale.combined(with: .opacity))
            } else {
                let empty = model.tutorDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                Button { model.askTutor() } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 22))
                        .foregroundStyle(empty ? AnyShapeStyle(HierarchicalShapeStyle.tertiary) : AnyShapeStyle(Theme.accent))
                }
                .disabled(empty || !model.hasAPIKey)
                .keyboardShortcut(.return, modifiers: .command)
                .help("Send (Return)")
                .transition(.scale.combined(with: .opacity))
            }
        }
        .buttonStyle(.borderless)
        .font(.body)
        .padding(.leading, 14)
        .padding(.trailing, 6)
        .padding(.vertical, 8)
        .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(inputFocused ? Theme.accent.opacity(0.6) : Theme.Palette.border, lineWidth: 1)
        }
        .padding(12)
        .animation(Theme.Motion.snappy, value: model.tutor.isAnswering)
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
                    .foregroundStyle(Theme.Palette.text)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Theme.Palette.selection, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .padding(.leading, 40)
            } else if message.isError {
                Label(message.text, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(Theme.fail)
                    .padding(10)
                    .background(Theme.fail.opacity(0.08), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            } else if message.text.isEmpty && streaming {
                TypingDots()
            } else {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "asterisk")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(Theme.accent)
                        .frame(width: 20, height: 20)
                    MarkdownView(source: message.text)
                }
            }
            if let note = message.note {
                Text(note).font(.caption).foregroundStyle(Theme.Palette.secondaryText)
            }
        }
    }
}

/// Three dots that bounce while the tutor starts its answer.
private struct TypingDots: View {
    var body: some View {
        TimelineView(.animation) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            HStack(spacing: 4) {
                ForEach(0..<3, id: \.self) { i in
                    Circle()
                        .fill(Color.secondary)
                        .frame(width: 6, height: 6)
                        .offset(y: -3 * max(0, sin((t * 6) - Double(i) * 0.7)))
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Color.primary.opacity(0.06), in: Capsule())
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
                        .foregroundStyle(Theme.Palette.secondaryText)
                }
            } header: {
                Text("Tutor")
            } footer: {
                Text("Get a key at console.anthropic.com. It's stored in your login Keychain and sent only to the Claude API. ANTHROPIC_API_KEY in the environment also works.")
                    .font(.caption)
                    .foregroundStyle(Theme.Palette.secondaryText)
            }
        }
        .formStyle(.grouped)
        .frame(width: 460)
        .padding()
    }
}
