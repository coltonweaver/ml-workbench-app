import Foundation

/// A request sent to the Python sidecar. One JSON object per line on its stdin.
/// See `protocol.md` for the full list of ops.
public struct SidecarRequest: Encodable, Equatable, Sendable {
    public enum Op: String, Encodable, Sendable {
        case execute
        case runFile = "run_file"
        case runTests = "run_tests"
        case interrupt
        case restart
        case shutdown
        case listLessons = "list_lessons"
        case openLesson = "open_lesson"
        case submitPrediction = "submit_prediction"
        case checkLesson = "check_lesson"
        case askTutor = "ask_tutor"
        case cancelTutor = "cancel_tutor"
    }

    public var id: String
    public var op: Op
    public var code: String?
    public var path: String?
    public var target: String?
    public var lesson: String?
    public var value: String?
    // ask_tutor
    public var question: String?
    public var history: [TutorTurn]?
    public var apiKey: String?
    public var tensor: JSONValue?

    public init(
        id: String, op: Op, code: String? = nil, path: String? = nil, target: String? = nil,
        lesson: String? = nil, value: String? = nil, question: String? = nil,
        history: [TutorTurn]? = nil, apiKey: String? = nil, tensor: JSONValue? = nil
    ) {
        self.id = id
        self.op = op
        self.code = code
        self.path = path
        self.target = target
        self.lesson = lesson
        self.value = value
        self.question = question
        self.history = history
        self.apiKey = apiKey
        self.tensor = tensor
    }

    enum CodingKeys: String, CodingKey {
        case id, op, code, path, target, lesson, value, question, history, tensor
        case apiKey = "api_key"
    }

    public func encodedLine() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(self)
        data.append(0x0A)
        return data
    }
}

/// An event from the sidecar. The wire format is flat, so this is one struct
/// with optional fields rather than an enum; `kind` says which fields are set.
public struct SidecarEvent: Decodable, Equatable, Sendable {
    public enum Kind: String, Decodable, Sendable {
        case ready
        case stream
        case executeResult = "execute_result"
        case displayData = "display_data"
        case updateDisplayData = "update_display_data"
        case error
        case testReport = "test_report"
        case lessons
        case lesson
        case tutorDelta = "tutor_delta"
        case tutorDone = "tutor_done"
        case done
    }

    public var kind: Kind
    public var id: String?

    // ready
    public var protocolVersion: Int?
    public var python: String?
    public var workspace: String?

    // stream
    public var name: String?
    public var text: String?

    // execute_result / display_data / update_display_data
    public var data: [String: JSONValue]?
    /// Set when the display can be updated in place.
    public var displayID: String?

    // error
    public var ename: String?
    public var evalue: String?
    public var traceback: [String]?

    // done
    public var status: String?

    // tutor_done (tutor_delta uses `text`)
    public var stopReason: String?
    public var model: String?
    public var note: String?

    // lessons / lesson
    public var modules: [ModuleSummary]?
    public var lesson: LessonDetail?

    // test_report (flattened into the event on the wire)
    public var outcome: String?
    public var passed: Int?
    public var failed: Int?
    public var tests: [TestCase]?
    public var output: String?

    enum CodingKeys: String, CodingKey {
        case kind = "type"
        case id
        case protocolVersion = "protocol"
        case python, workspace, name, text, data, ename, evalue, traceback, status
        case displayID = "display_id"
        case outcome, passed, failed, tests, output
        case modules, lesson
        case stopReason = "stop_reason"
        case model, note
    }

    /// `text/plain` of an execute_result or display_data, if any.
    public var plainText: String? {
        if case .string(let s)? = data?["text/plain"] { return s }
        return nil
    }

    public var testReport: TestReport? {
        guard kind == .testReport, let outcome else { return nil }
        return TestReport(
            outcome: outcome,
            passed: passed ?? 0,
            failed: failed ?? 0,
            tests: tests ?? [],
            output: output ?? ""
        )
    }
}

public struct TestCase: Decodable, Equatable, Sendable, Identifiable {
    public var nodeid: String
    public var outcome: String
    public var duration: Double
    public var message: String?

    public var id: String { nodeid }
    public var didPass: Bool { outcome == "passed" }
}

public struct TestReport: Equatable, Sendable {
    /// "passed", "failed", "no_tests", "timeout", "cancelled" or "error".
    public var outcome: String
    public var passed: Int
    public var failed: Int
    public var tests: [TestCase]
    /// pytest's own console output, for when the structured report isn't enough.
    public var output: String
}

/// One message of the tutor chat, as the learner saw it.
public struct TutorTurn: Codable, Equatable, Sendable {
    public enum Role: String, Codable, Sendable {
        case user, assistant
    }

    public var role: Role
    public var text: String

    public init(role: Role, text: String) {
        self.role = role
        self.text = text
    }
}

/// Minimal JSON tree for the `data` bundles the kernel returns.
public enum JSONValue: Codable, Equatable, Sendable {
    case string(String)
    /// Whole numbers stay integers so they round-trip into `Int` fields.
    case integer(Int)
    case number(Double)
    case bool(Bool)
    case null
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let v = try? c.decode(Bool.self) {
            self = .bool(v)
        } else if let v = try? c.decode(Int.self) {
            self = .integer(v)
        } else if let v = try? c.decode(Double.self) {
            self = .number(v)
        } else if let v = try? c.decode(String.self) {
            self = .string(v)
        } else if let v = try? c.decode([JSONValue].self) {
            self = .array(v)
        } else {
            self = .object(try c.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .integer(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .null: try c.encodeNil()
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }
}

/// Splits a byte stream into newline-terminated lines, holding any partial
/// line until the rest arrives.
public struct LineBuffer: Sendable {
    private var pending = Data()

    public init() {}

    public mutating func append(_ chunk: Data) -> [Data] {
        pending.append(chunk)
        var lines: [Data] = []
        while let newline = pending.firstIndex(of: 0x0A) {
            let line = pending[pending.startIndex..<newline]
            if !line.isEmpty { lines.append(Data(line)) }
            pending = Data(pending[pending.index(after: newline)...])
        }
        return lines
    }
}

// MARK: Lessons

public struct ModuleSummary: Decodable, Equatable, Sendable, Identifiable {
    public var id: String
    public var number: Int
    public var title: String
    public var summary: String
    public var lessons: [LessonSummary]
}

public struct LessonSummary: Decodable, Equatable, Sendable, Identifiable {
    public var id: String
    public var title: String
    /// "new", "in_progress" or "passed".
    public var status: String
}

public struct LessonDetail: Decodable, Equatable, Sendable, Identifiable {
    public struct Predict: Decodable, Equatable, Sendable {
        /// "choice", "number" or "text".
        public var kind: String
        public var prompt: String
        public var options: [String]?
    }

    public struct Prediction: Decodable, Equatable, Sendable {
        public var value: String
        public var submittedAt: String

        enum CodingKeys: String, CodingKey {
            case value
            case submittedAt = "submitted_at"
        }
    }

    public struct Feedback: Decodable, Equatable, Sendable {
        /// nil for free-text predictions, which aren't graded.
        public var correct: Bool?
        public var yours: String
        public var answer: String?
    }

    public struct Progress: Decodable, Equatable, Sendable {
        public var status: String
        public var attempts: Int
        public var prediction: Prediction?
        public var feedback: Feedback?
    }

    public var id: String
    public var title: String
    public var moduleID: String
    public var moduleTitle: String
    /// Workspace-relative path of the file the learner edits.
    public var file: String
    public var concepts: [String]
    public var hook: String
    public var build: String
    /// Only sent once the lesson is passed.
    public var explain: String?
    public var stretch: String?
    public var predict: Predict
    public var progress: Progress

    public var isPassed: Bool { progress.status == "passed" }
    public var hasPrediction: Bool { progress.prediction != nil }

    enum CodingKeys: String, CodingKey {
        case id, title, file, concepts, hook, build, explain, stretch, predict, progress
        case moduleID = "module_id"
        case moduleTitle = "module_title"
    }
}
