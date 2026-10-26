import Foundation
import SwiftUI
import WorkbenchCore

struct InstrumentItem: Identifiable, Equatable {
    /// The display id, so a later update replaces the same widget.
    let id: String
    var instrument: Instrument
    /// Tensor summaries for "Explain this tensor".
    var explain: [JSONValue] = []
}

struct OutputChunk: Identifiable, Equatable {
    enum Style { case stdout, stderr, result, error, info }
    let id = UUID()
    var style: Style
    var text: String
}

enum SidebarItem: Hashable {
    /// The review queue and prediction score.
    case home
    case lesson(String)
    case file(String)
}

enum KernelState: Equatable {
    case stopped
    case starting
    case idle
    case busy
    case failed(String)

    var label: String {
        switch self {
        case .stopped: "Stopped"
        case .starting: "Starting kernel…"
        case .idle: "Ready"
        case .busy: "Running…"
        case .failed: "Kernel failed"
        }
    }
}

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var workspace: Workspace
    @Published private(set) var files: [String] = []
    @Published var selectedFile: String? {
        didSet { if selectedFile != oldValue { loadSelected() } }
    }
    @Published var editorText = "" {
        didSet { isDirty = editorText != savedText }
    }
    @Published private(set) var isDirty = false
    @Published private(set) var output: [OutputChunk] = []
    @Published private(set) var testReport: TestReport?
    /// Newest first.
    @Published private(set) var instruments: [InstrumentItem] = []
    @Published private(set) var kernelState: KernelState = .stopped
    @Published var outputTab: OutputTab = .output
    @Published var errorMessage: String?

    @Published private(set) var modules: [ModuleSummary] = []
    @Published private(set) var lesson: LessonDetail?
    /// What the learner has typed or picked for the open lesson's prediction.
    /// For a choice prediction this is the option's index.
    @Published var predictionDraft = ""
    /// How sure the learner is of that prediction, if they said.
    @Published var predictionConfidence: Double?

    // Review queue and prediction score
    @Published private(set) var review: ReviewState?
    /// Typed or picked answers, by warm-up id.
    @Published var warmupDrafts: [String: String] = [:]
    /// The warm-ups answered this session, newest first, so the home screen
    /// can keep showing the result after the queue moves on.
    @Published private(set) var warmupResults: [WarmupResult] = []
    @Published var selection: SidebarItem? {
        didSet { if selection != oldValue { selectionChanged() } }
    }

    enum OutputTab: Hashable { case output, tests, instruments }

    // Tutor
    @Published private(set) var tutor = TutorChat()
    @Published var tutorDraft = ""
    @Published var showTutor = false
    @Published private(set) var hasAPIKey = false

    func clearInstruments() {
        instruments.removeAll()
    }

    func clearOutput() {
        output.removeAll()
    }

    private var savedText = ""
    private var client: SidecarClient?
    private var eventTask: Task<Void, Never>?
    private var pending: Set<String> = []
    /// Completion requests waiting for their answer, by request id.
    private var completionWaiters: [String: CheckedContinuation<CompletionResult?, Never>] = [:]
    /// Completion requests whose `done` hasn't arrived yet.
    private var completionRequests: Set<String> = []

    init(workspaceRoot: URL = Workspace.defaultRoot()) {
        workspace = Workspace(root: workspaceRoot)
    }

    // MARK: Lifecycle

    func start() {
        refreshAPIKeyStatus()
        do {
            try workspace.prepare()
            refreshFiles()
            if selection == nil { selection = .home }
        } catch {
            errorMessage = "Couldn't prepare workspace at \(workspace.root.path): \(error.localizedDescription)"
        }
        startKernel()
    }

    func startKernel() {
        guard let uv = SidecarLaunch.findUV() else {
            kernelState = .failed("uv not found")
            errorMessage = "Couldn't find uv. Install it with `brew install uv` (or see docs.astral.sh/uv), then restart the kernel."
            return
        }
        guard let project = SidecarLaunch.findPythonProject() else {
            kernelState = .failed("python project not found")
            errorMessage = "Couldn't find the bundled python/ project. Set WORKBENCH_PYTHON_PROJECT to its path."
            return
        }
        stopKernel()
        let launch = SidecarLaunch(
            uv: uv, pythonProject: project, workspace: workspace.root,
            content: SidecarLaunch.findContent(),
            stateDirectory: SidecarLaunch.stateDirectory(for: project)
        )
        let client = SidecarClient(launch: launch)
        client.onExit = { [weak self] error in
            Task { @MainActor in
                self?.kernelState = .failed("exited")
                self?.append(.error, error.localizedDescription + "\n")
            }
        }
        self.client = client
        kernelState = .starting
        append(.info, "Starting Python (the first launch installs packages and can take a minute)…\n")
        do {
            try client.start()
        } catch {
            kernelState = .failed(error.localizedDescription)
            errorMessage = "Couldn't launch the sidecar: \(error.localizedDescription)"
            return
        }
        eventTask = Task { [weak self] in
            for await event in client.events {
                self?.handle(event)
            }
        }
    }

    func stopKernel() {
        eventTask?.cancel()
        eventTask = nil
        client?.onExit = nil
        client?.shutdown()
        client = nil
        pending.removeAll()
        for id in Array(completionWaiters.keys) { finishCompletion(id, nil) }
        completionRequests.removeAll()
        kernelState = .stopped
    }

    // MARK: Selection

    private func selectionChanged() {
        save()
        switch selection {
        case .home:
            lesson = nil
            if let client { submit { try client.getReview() } }
        case .file(let path):
            lesson = nil
            selectedFile = path
        case .lesson(let id):
            predictionDraft = ""
            predictionConfidence = nil
            guard let client else { return }
            // The sidecar answers with a `lesson` event; handle() opens its file.
            do {
                try client.openLesson(id)
            } catch {
                append(.error, "Couldn't reach the sidecar: \(error.localizedDescription)\n")
            }
        case nil:
            lesson = nil
        }
    }

    // MARK: Files

    func refreshFiles() {
        files = (try? workspace.pythonFiles()) ?? []
    }

    private func loadSelected() {
        guard let path = selectedFile else {
            savedText = ""
            editorText = ""
            isDirty = false
            return
        }
        do {
            let text = try workspace.read(path)
            savedText = text
            editorText = text
            isDirty = false
        } catch {
            errorMessage = "Couldn't open \(path): \(error.localizedDescription)"
        }
    }

    func save() {
        guard let path = selectedFile, isDirty else { return }
        do {
            try workspace.write(path, contents: editorText)
            savedText = editorText
            isDirty = false
        } catch {
            errorMessage = "Couldn't save \(path): \(error.localizedDescription)"
        }
    }

    /// Re-reads the open file if it changed on disk and has no unsaved edits,
    /// so edits made in another editor show up.
    func reloadIfChangedOnDisk() {
        refreshFiles()
        guard let path = selectedFile, !isDirty, let text = try? workspace.read(path), text != savedText else { return }
        savedText = text
        editorText = text
        isDirty = false
    }

    // MARK: Actions

    var canRun: Bool { selectedFile != nil && kernelState == .idle && !needsPrediction }

    /// Problem first: in a lesson, code doesn't run until a prediction is in.
    var needsPrediction: Bool {
        guard let lesson else { return false }
        return !lesson.hasPrediction && selectedFile == lesson.file
    }

    var canCheck: Bool { lesson?.hasPrediction == true && kernelState == .idle }

    func submitPrediction() {
        guard let lesson, let client else { return }
        let value = predictionDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return }
        submit { try client.submitPrediction(lesson: lesson.id, value: value, confidence: predictionConfidence) }
    }

    func answer(_ warmup: Warmup) {
        guard let client else { return }
        let value = (warmupDrafts[warmup.id] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return }
        submit { try client.answerWarmup(warmup.id, value: value) }
    }

    func check() {
        guard let lesson, let client, canCheck else { return }
        save()
        testReport = nil
        outputTab = .tests
        submit { try client.checkLesson(lesson.id) }
    }

    func run() {
        guard let path = selectedFile, let client else { return }
        save()
        output.removeAll()
        outputTab = .output
        submit { try client.runFile(path) }
    }

    func test() {
        guard let path = selectedFile, let client else { return }
        save()
        let target = workspace.testTarget(for: path)
        testReport = nil
        outputTab = .tests
        append(.info, "Testing \(target)…\n")
        submit { try client.runTests(target) }
    }

    func interrupt() {
        try? client?.interrupt()
    }

    func restartKernel() {
        guard let client else { return startKernel() }
        output.removeAll()
        append(.info, "Restarting kernel…\n")
        submit { try client.restart() }
    }

    private func submit(_ send: () throws -> String) {
        do {
            pending.insert(try send())
            kernelState = .busy
        } catch {
            append(.error, "Couldn't reach the sidecar: \(error.localizedDescription)\n")
        }
    }

    // MARK: Tutor

    /// Asks the tutor `question`, or what's in the draft. With `tensor`, the
    /// question is about that tensor summary.
    func askTutor(_ question: String? = nil, tensor: JSONValue? = nil) {
        let text = (question ?? tutorDraft).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !tutor.isAnswering else { return }
        showTutor = true
        guard let client else {
            errorMessage = "The Python sidecar isn't running, so the tutor can't be reached. Restart the kernel."
            return
        }
        let key: String?
        do {
            key = try KeychainItem.apiKey.read()
        } catch {
            errorMessage = error.localizedDescription
            return
        }
        // The open lesson's file goes as typed, saved or not.
        let code: String? = if let lesson {
            selectedFile == lesson.file ? editorText : nil
        } else {
            selectedFile != nil ? editorText : nil
        }
        let history = tutor.history
        do {
            let id = try client.askTutor(
                question: text, history: history, apiKey: key, lesson: lesson?.id, code: code, tensor: tensor
            )
            tutor.ask(text, request: id)
            if question == nil { tutorDraft = "" }
        } catch {
            errorMessage = "Couldn't reach the sidecar: \(error.localizedDescription)"
        }
    }

    /// Re-runs an attention viewer's model on a new prompt, in the kernel.
    func rerun(attention: AttentionPayload, prompt: String) {
        guard let client, let code = attention.rerunCode(prompt: prompt) else { return }
        submit { try client.execute(code) }
    }

    func explain(_ summary: JSONValue) {
        askTutor("Explain this tensor: \(summary.name ?? "tensor")", tensor: summary)
    }

    func stopTutor() {
        try? client?.cancelTutor()
    }

    func clearTutor() {
        stopTutor()
        tutor.clear()
    }

    func refreshAPIKeyStatus() {
        let stored = (try? KeychainItem.apiKey.read()).flatMap { $0 }
        hasAPIKey = !(stored ?? "").isEmpty || ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"] != nil
    }

    func saveAPIKey(_ key: String) {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            if trimmed.isEmpty {
                try KeychainItem.apiKey.delete()
            } else {
                try KeychainItem.apiKey.save(trimmed)
            }
        } catch {
            errorMessage = error.localizedDescription
        }
        refreshAPIKeyStatus()
    }

    private func handleTutor(_ event: SidecarEvent, request: String) {
        switch event.kind {
        case .tutorDelta:
            tutor.receive(delta: event.text ?? "", request: request)
        case .tutorDone:
            tutor.finish(request: request, stopReason: event.stopReason, note: event.note)
        case .error:
            tutor.fail(request: request, message: event.evalue ?? "The tutor failed.")
        case .done:
            tutor.end(request: request)
        default:
            break
        }
    }

    // MARK: Completions

    struct CompletionResult {
        var completions: [Completion]
        var signature: CallSignature?
    }

    /// Asks the sidecar for completions at a position in the editor's text.
    /// nil when the sidecar isn't running or a newer request overtook this one.
    func completions(code: String, line: Int, column: Int) async -> CompletionResult? {
        guard let client else { return nil }
        let path = selectedFile
        return await withCheckedContinuation { continuation in
            do {
                let id = try client.complete(code: code, path: path, line: line, column: column)
                completionWaiters[id] = continuation
                completionRequests.insert(id)
            } catch {
                continuation.resume(returning: nil)
            }
        }
    }

    private func finishCompletion(_ id: String, _ result: CompletionResult?) {
        completionWaiters.removeValue(forKey: id)?.resume(returning: result)
    }

    // MARK: Events

    private func handle(_ event: SidecarEvent) {
        if let id = event.id, id == tutor.activeRequest {
            handleTutor(event, request: id)
            return
        }
        if let id = event.id, completionRequests.contains(id) {
            switch event.kind {
            case .completions:
                finishCompletion(id, CompletionResult(completions: event.completions ?? [],
                                                      signature: event.signatures?.first))
            case .done:
                completionRequests.remove(id)
                finishCompletion(id, nil)
            case .error:
                finishCompletion(id, nil)
            default:
                break
            }
            return
        }
        switch event.kind {
        case .ready:
            kernelState = .idle
            append(.info, "Python \(event.python ?? "?") ready in \(event.workspace ?? workspace.root.path)\n")
            if let client {
                submit { try client.listLessons() }
                submit { try client.getReview() }
                if case .lesson(let id)? = selection { submit { try client.openLesson(id) } }
            }
        case .lessons:
            modules = event.modules ?? []
        case .review:
            guard let state = event.review else { break }
            review = state
            if let result = state.result {
                warmupResults.removeAll { $0.id == result.id }
                warmupResults.insert(result, at: 0)
                warmupDrafts[result.id] = nil
            }
        case .lesson:
            guard let detail = event.lesson, selection == .lesson(detail.id) else { break }
            lesson = detail
            if selectedFile != detail.file {
                refreshFiles()
                selectedFile = detail.file
            }
        case .stream:
            append(event.name == "stderr" ? .stderr : .stdout, event.text ?? "")
        case .executeResult, .displayData, .updateDisplayData:
            if let data = event.data, let instrument = Instrument(data: data) {
                show(instrument, explain: Instrument.explainSummaries(data: data),
                     displayID: event.displayID, isUpdate: event.kind == .updateDisplayData)
            } else if event.kind != .updateDisplayData, let text = event.plainText {
                append(.result, text + "\n")
            }
        case .error where event.ename == "LessonError":
            errorMessage = event.evalue
        case .error:
            var text = "\(event.ename ?? "Error"): \(event.evalue ?? "")\n"
            if let tb = event.traceback, !tb.isEmpty {
                text = tb.map(stripANSI).joined(separator: "\n") + "\n"
            }
            append(.error, text)
        case .testReport:
            testReport = event.testReport
        case .tutorDelta, .tutorDone, .completions:
            break  // a stopped answer's stragglers, or a completion nobody waits for
        case .done:
            if let id = event.id { pending.remove(id) }
            if pending.isEmpty, kernelState == .busy { kernelState = .idle }
        }
    }

    private func show(_ instrument: Instrument, explain: [JSONValue], displayID: String?, isUpdate: Bool) {
        let id = displayID ?? UUID().uuidString
        if let index = instruments.firstIndex(where: { $0.id == id }) {
            instruments[index].instrument = instrument
            instruments[index].explain = explain
        } else if !isUpdate {
            instruments.insert(InstrumentItem(id: id, instrument: instrument, explain: explain), at: 0)
            outputTab = .instruments
        }
    }

    private func append(_ style: OutputChunk.Style, _ text: String) {
        if let last = output.last, last.style == style {
            output[output.count - 1].text += text
        } else {
            output.append(OutputChunk(style: style, text: text))
        }
    }
}

/// IPython colors its tracebacks; the output pane shows plain text for now.
func stripANSI(_ s: String) -> String {
    s.replacingOccurrences(of: "\u{1B}\\[[0-9;]*m", with: "", options: .regularExpression)
}
