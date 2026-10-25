import Foundation
import SwiftUI
import WorkbenchCore

struct OutputChunk: Identifiable, Equatable {
    enum Style { case stdout, stderr, result, error, info }
    let id = UUID()
    var style: Style
    var text: String
}

enum SidebarItem: Hashable {
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
    @Published private(set) var kernelState: KernelState = .stopped
    @Published var outputTab: OutputTab = .output
    @Published var errorMessage: String?

    @Published private(set) var modules: [ModuleSummary] = []
    @Published private(set) var lesson: LessonDetail?
    /// What the learner has typed or picked for the open lesson's prediction.
    /// For a choice prediction this is the option's index.
    @Published var predictionDraft = ""
    @Published var selection: SidebarItem? {
        didSet { if selection != oldValue { selectionChanged() } }
    }

    enum OutputTab: Hashable { case output, tests }

    private var savedText = ""
    private var client: SidecarClient?
    private var eventTask: Task<Void, Never>?
    private var pending: Set<String> = []

    init(workspaceRoot: URL = Workspace.defaultRoot()) {
        workspace = Workspace(root: workspaceRoot)
    }

    // MARK: Lifecycle

    func start() {
        do {
            try workspace.prepare()
            refreshFiles()
            if selection == nil, let first = files.first { selection = .file(first) }
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
            content: SidecarLaunch.findContent()
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
        kernelState = .stopped
    }

    // MARK: Selection

    private func selectionChanged() {
        save()
        switch selection {
        case .file(let path):
            lesson = nil
            selectedFile = path
        case .lesson(let id):
            predictionDraft = ""
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
        submit { try client.submitPrediction(lesson: lesson.id, value: value) }
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

    // MARK: Events

    private func handle(_ event: SidecarEvent) {
        switch event.kind {
        case .ready:
            kernelState = .idle
            append(.info, "Python \(event.python ?? "?") ready in \(event.workspace ?? workspace.root.path)\n")
            if let client {
                submit { try client.listLessons() }
                if case .lesson(let id)? = selection { submit { try client.openLesson(id) } }
            }
        case .lessons:
            modules = event.modules ?? []
        case .lesson:
            guard let detail = event.lesson, selection == .lesson(detail.id) else { break }
            lesson = detail
            if selectedFile != detail.file {
                refreshFiles()
                selectedFile = detail.file
            }
        case .stream:
            append(event.name == "stderr" ? .stderr : .stdout, event.text ?? "")
        case .executeResult, .displayData:
            if let text = event.plainText { append(.result, text + "\n") }
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
        case .done:
            if let id = event.id { pending.remove(id) }
            if pending.isEmpty, kernelState == .busy { kernelState = .idle }
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
