import XCTest
@testable import WorkbenchCore

final class ProtocolTests: XCTestCase {
    func testRequestEncodesOneLineWithSnakeCaseOp() throws {
        let line = try SidecarRequest(id: "r1", op: .runFile, path: "m01/engine.py").encodedLine()
        let text = String(decoding: line, as: UTF8.self)
        XCTAssertEqual(text, #"{"id":"r1","op":"run_file","path":"m01/engine.py"}"# + "\n")
    }

    func testDecodesStreamEvent() throws {
        let json = #"{"type":"stream","id":"r2","name":"stdout","text":"hi\n"}"#
        let event = try JSONDecoder().decode(SidecarEvent.self, from: Data(json.utf8))
        XCTAssertEqual(event.kind, .stream)
        XCTAssertEqual(event.id, "r2")
        XCTAssertEqual(event.text, "hi\n")
    }

    func testDecodesExecuteResultPlainText() throws {
        let json = #"{"type":"execute_result","id":"r3","data":{"text/plain":"42","application/json":{"a":[1,null]}}}"#
        let event = try JSONDecoder().decode(SidecarEvent.self, from: Data(json.utf8))
        XCTAssertEqual(event.plainText, "42")
    }

    func testDecodesTestReport() throws {
        let json = """
        {"type":"test_report","id":"r4","outcome":"failed","exit_code":1,"passed":1,"failed":1,
         "tests":[{"nodeid":"test_a.py::test_ok","outcome":"passed","duration":0.01,"message":null},
                  {"nodeid":"test_a.py::test_bad","outcome":"failed","duration":0.02,"message":"AssertionError: nope"}],
         "output":"1 failed, 1 passed"}
        """
        let event = try JSONDecoder().decode(SidecarEvent.self, from: Data(json.utf8))
        let report = try XCTUnwrap(event.testReport)
        XCTAssertEqual(report.outcome, "failed")
        XCTAssertEqual(report.tests.map(\.didPass), [true, false])
        XCTAssertEqual(report.tests[1].message, "AssertionError: nope")
    }

    func testReadyEventCarriesProtocolVersion() throws {
        let json = #"{"type":"ready","protocol":1,"python":"3.12.7","workspace":"/tmp/ws"}"#
        let event = try JSONDecoder().decode(SidecarEvent.self, from: Data(json.utf8))
        XCTAssertEqual(event.protocolVersion, 1)
    }

    func testLineBufferHoldsPartialLines() {
        var buffer = LineBuffer()
        XCTAssertEqual(buffer.append(Data("{\"a\":".utf8)), [])
        XCTAssertEqual(buffer.append(Data("1}\n{\"b\":2}\n{\"c\"".utf8)).map { String(decoding: $0, as: UTF8.self) },
                       [#"{"a":1}"#, #"{"b":2}"#])
        XCTAssertEqual(buffer.append(Data(":3}\n".utf8)).map { String(decoding: $0, as: UTF8.self) },
                       [#"{"c":3}"#])
    }

    func testCompleteRequestEncodes() throws {
        let request = SidecarRequest(id: "c1", op: .complete, code: "math.fl", path: "m01/slope.py",
                                     line: 3, column: 7)
        let line = String(decoding: try request.encodedLine(), as: UTF8.self)
        XCTAssertEqual(line, #"{"code":"math.fl","column":7,"id":"c1","line":3,"op":"complete","path":"m01/slope.py"}"# + "\n")
    }

    func testDecodesCompletions() throws {
        let json = """
        {"type":"completions","id":"c1",
         "completions":[{"name":"floor","complete":"oor","type":"function","signature":"floor(x, /)","doc":"Return the floor."},
                        {"name":"fmod","complete":"","type":"function","signature":null,"doc":null}],
         "signatures":[{"name":"slope","params":["f","x","h=1e-5"],"index":1,"doc":null}]}
        """
        let event = try JSONDecoder().decode(SidecarEvent.self, from: Data(json.utf8))
        XCTAssertEqual(event.kind, .completions)
        XCTAssertEqual(event.completions?.map(\.name), ["floor", "fmod"])
        XCTAssertEqual(event.completions?.first?.complete, "oor")
        XCTAssertEqual(event.signatures?.first?.index, 1)
        XCTAssertEqual(event.signatures?.first?.params.count, 3)
    }
}
