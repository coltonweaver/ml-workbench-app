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
    }

    public var id: String
    public var op: Op
    public var code: String?
    public var path: String?
    public var target: String?

    public init(id: String, op: Op, code: String? = nil, path: String? = nil, target: String? = nil) {
        self.id = id
        self.op = op
        self.code = code
        self.path = path
        self.target = target
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
        case error
        case testReport = "test_report"
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

    // execute_result / display_data
    public var data: [String: JSONValue]?

    // error
    public var ename: String?
    public var evalue: String?
    public var traceback: [String]?

    // done
    public var status: String?

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
        case outcome, passed, failed, tests, output
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

/// Minimal JSON tree for the `data` bundles the kernel returns.
public enum JSONValue: Decodable, Equatable, Sendable {
    case string(String)
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
