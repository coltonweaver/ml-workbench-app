import Foundation

/// How to launch the sidecar: which `uv`, which Python project, which workspace.
public struct SidecarLaunch: Sendable {
    public var uv: URL
    public var pythonProject: URL
    public var workspace: URL
    /// The lesson content folder; lessons are unavailable without it.
    public var content: URL?

    public init(uv: URL, pythonProject: URL, workspace: URL, content: URL? = nil) {
        self.uv = uv
        self.pythonProject = pythonProject
        self.workspace = workspace
        self.content = content
    }

    public var arguments: [String] {
        var args = [
            "run", "--quiet", "--project", pythonProject.path,
            "python", "-m", "workbench_sidecar", "--workspace", workspace.path,
        ]
        if let content {
            args += ["--content", content.path]
        }
        return args
    }

    /// GUI apps don't inherit the login shell's PATH, so look in the usual
    /// install locations as well. `WORKBENCH_UV` overrides the search.
    public static func findUV(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL? {
        let fm = FileManager.default
        if let explicit = environment["WORKBENCH_UV"], fm.isExecutableFile(atPath: explicit) {
            return URL(fileURLWithPath: explicit)
        }
        let home = fm.homeDirectoryForCurrentUser.path
        var candidates = (environment["PATH"] ?? "").split(separator: ":").map { "\($0)/uv" }
        candidates += [
            "\(home)/.local/bin/uv",
            "\(home)/.cargo/bin/uv",
            "/opt/homebrew/bin/uv",
            "/usr/local/bin/uv",
        ]
        return candidates.first(where: fm.isExecutableFile(atPath:)).map(URL.init(fileURLWithPath:))
    }

    /// The `python/` project: `WORKBENCH_PYTHON_PROJECT`, then the app bundle's
    /// resources, then the source checkout (for `swift run`).
    public static func findPythonProject(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        sourceFile: String = #filePath
    ) -> URL? {
        findBundled("python", marker: "pyproject.toml", envKey: "WORKBENCH_PYTHON_PROJECT",
                    environment: environment, sourceFile: sourceFile)
    }

    /// The lesson `content/` folder, searched the same way (`WORKBENCH_CONTENT`).
    public static func findContent(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        sourceFile: String = #filePath
    ) -> URL? {
        findBundled("content", marker: "README.md", envKey: "WORKBENCH_CONTENT",
                    environment: environment, sourceFile: sourceFile)
    }

    static func findBundled(
        _ name: String, marker: String, envKey: String,
        environment: [String: String], sourceFile: String
    ) -> URL? {
        let fm = FileManager.default
        var candidates: [URL] = []
        if let explicit = environment[envKey] {
            candidates.append(URL(fileURLWithPath: explicit))
        }
        if let resources = Bundle.main.resourceURL {
            candidates.append(resources.appendingPathComponent(name))
        }
        // Sources/WorkbenchCore/SidecarClient.swift -> repo root
        candidates.append(
            URL(fileURLWithPath: sourceFile)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent(name)
        )
        return candidates.first { fm.fileExists(atPath: $0.appendingPathComponent(marker).path) }
    }
}

public enum SidecarError: Error, LocalizedError {
    case notRunning
    case exited(status: Int32, stderr: String)

    public var errorDescription: String? {
        switch self {
        case .notRunning:
            return "The Python sidecar isn't running."
        case .exited(let status, let stderr):
            return "The Python sidecar exited with status \(status).\n\(stderr)"
        }
    }
}

/// Owns the sidecar process. Requests go in through the typed methods; every
/// event the sidecar emits comes out of `events`, in order.
public final class SidecarClient: @unchecked Sendable {
    public let events: AsyncStream<SidecarEvent>

    private let launch: SidecarLaunch
    private let continuation: AsyncStream<SidecarEvent>.Continuation
    private let lock = NSLock()
    private var process: Process?
    private var stdin: FileHandle?
    private var buffer = LineBuffer()
    private var stderrText = ""
    private var nextID = 0

    public init(launch: SidecarLaunch) {
        self.launch = launch
        let stream = AsyncStream.makeStream(of: SidecarEvent.self)
        events = stream.stream
        continuation = stream.continuation
    }

    /// Exit status and captured stderr, delivered when the process ends.
    public var onExit: (@Sendable (SidecarError) -> Void)?

    public func start() throws {
        let process = Process()
        process.executableURL = launch.uv
        process.arguments = launch.arguments
        process.currentDirectoryURL = launch.workspace

        let inPipe = Pipe(), outPipe = Pipe(), errPipe = Pipe()
        process.standardInput = inPipe
        process.standardOutput = outPipe
        process.standardError = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.receive(handle.availableData)
        }
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard let self, !chunk.isEmpty else { return }
            self.lock.withLock { self.stderrText += String(decoding: chunk, as: UTF8.self) }
        }
        process.terminationHandler = { [weak self] proc in
            guard let self else { return }
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            let stderr = self.lock.withLock { self.stderrText }
            if proc.terminationStatus != 0 {
                self.onExit?(.exited(status: proc.terminationStatus, stderr: stderr))
            }
            self.continuation.finish()
        }

        try FileManager.default.createDirectory(at: launch.workspace, withIntermediateDirectories: true)
        try process.run()
        lock.withLock {
            self.process = process
            self.stdin = inPipe.fileHandleForWriting
        }
    }

    @discardableResult
    public func execute(_ code: String) throws -> String {
        try send(.execute) { $0.code = code }
    }

    /// Runs a workspace-relative file in the kernel with `%run`.
    @discardableResult
    public func runFile(_ path: String) throws -> String {
        try send(.runFile) { $0.path = path }
    }

    /// Runs pytest on a workspace-relative file or directory, out of process.
    @discardableResult
    public func runTests(_ target: String) throws -> String {
        try send(.runTests) { $0.target = target }
    }

    @discardableResult
    public func restart() throws -> String {
        try send(.restart)
    }

    @discardableResult
    public func listLessons() throws -> String {
        try send(.listLessons)
    }

    /// Copies the lesson's stub into the workspace if the file isn't there yet.
    @discardableResult
    public func openLesson(_ id: String) throws -> String {
        try send(.openLesson) { $0.lesson = id }
    }

    @discardableResult
    public func submitPrediction(lesson id: String, value: String, confidence: Double? = nil) throws -> String {
        try send(.submitPrediction) {
            $0.lesson = id
            $0.value = value
            $0.confidence = confidence
        }
    }

    /// The review queue and prediction score; answers with a `review` event.
    @discardableResult
    public func getReview() throws -> String {
        try send(.getReview)
    }

    @discardableResult
    public func answerWarmup(_ id: String, value: String) throws -> String {
        try send(.answerWarmup) {
            $0.warmup = id
            $0.value = value
        }
    }

    /// Runs the lesson's hidden tests and records the result in progress.json.
    @discardableResult
    public func checkLesson(_ id: String) throws -> String {
        try send(.checkLesson) { $0.lesson = id }
    }

    /// Asks the tutor. `code` is the editor's current text (it may not be
    /// saved yet); `tensor` is set by "Explain this tensor". The answer
    /// streams back as `tutor_delta` events, then `tutor_done`.
    @discardableResult
    public func askTutor(
        question: String, history: [TutorTurn], apiKey: String?, lesson: String?,
        code: String?, tensor: JSONValue? = nil
    ) throws -> String {
        try send(.askTutor) {
            $0.question = question
            $0.history = history
            $0.apiKey = apiKey
            $0.lesson = lesson
            $0.code = code
            $0.tensor = tensor
        }
    }

    /// Completions and the enclosing call's signature at a position in
    /// `code`. `line` is 1-based, `column` 0-based in Unicode scalars. Answers
    /// with one `completions` event; a request overtaken by a newer one is
    /// answered empty.
    @discardableResult
    public func complete(code: String, path: String?, line: Int, column: Int) throws -> String {
        try send(.complete) {
            $0.code = code
            $0.path = path
            $0.line = line
            $0.column = column
        }
    }

    public func cancelTutor() throws {
        try send(.cancelTutor)
    }

    public func interrupt() throws {
        try send(.interrupt)
    }

    /// Asks the sidecar to stop the kernel and exit; kills it if it hasn't
    /// gone within `timeout`.
    public func shutdown(timeout: TimeInterval = 5) {
        _ = try? send(.shutdown)
        let process = lock.withLock { self.process }
        guard let process else { return }
        DispatchQueue.global().asyncAfter(deadline: .now() + timeout) {
            if process.isRunning { process.terminate() }
        }
    }

    @discardableResult
    private func send(_ op: SidecarRequest.Op, configure: (inout SidecarRequest) -> Void = { _ in }) throws -> String {
        let (handle, id): (FileHandle?, String) = lock.withLock {
            nextID += 1
            return (stdin, "r\(nextID)")
        }
        guard let handle else { throw SidecarError.notRunning }
        var request = SidecarRequest(id: id, op: op)
        configure(&request)
        try handle.write(contentsOf: request.encodedLine())
        return id
    }

    private func receive(_ chunk: Data) {
        guard !chunk.isEmpty else { return }
        let lines = lock.withLock { buffer.append(chunk) }
        let decoder = JSONDecoder()
        for line in lines {
            if let event = try? decoder.decode(SidecarEvent.self, from: line) {
                continuation.yield(event)
            }
        }
    }
}
