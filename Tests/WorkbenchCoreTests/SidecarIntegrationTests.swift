import XCTest
@testable import WorkbenchCore

/// Drives the real Python sidecar through `uv`. Skipped unless
/// `WORKBENCH_INTEGRATION=1`, since it needs uv and a network for the first sync.
final class SidecarIntegrationTests: XCTestCase {
    func testRunFileAndTestsThroughRealSidecar() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["WORKBENCH_INTEGRATION"] == "1",
                          "set WORKBENCH_INTEGRATION=1 to run")
        let uv = try XCTUnwrap(SidecarLaunch.findUV(), "uv not found")
        let project = try XCTUnwrap(SidecarLaunch.findPythonProject(), "python/ project not found")

        let root = FileManager.default.temporaryDirectory.appendingPathComponent("ws-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let ws = Workspace(root: root)
        try ws.prepare(initGit: false)

        let content = try XCTUnwrap(SidecarLaunch.findContent(), "content/ not found")
        let client = SidecarClient(launch: SidecarLaunch(
            uv: uv, pythonProject: project, workspace: root, content: content
        ))
        try client.start()
        defer { client.shutdown() }

        var events = client.events.makeAsyncIterator()
        func collect(until id: String) async throws -> [SidecarEvent] {
            var out: [SidecarEvent] = []
            while let event = await events.next() {
                guard event.id == id else { continue }
                out.append(event)
                if event.kind == .done { return out }
            }
            XCTFail("sidecar exited early")
            return out
        }

        let ready = await events.next()
        XCTAssertEqual(ready?.kind, .ready)

        let run = try await collect(until: client.runFile("scratch.py"))
        let stdout = run.filter { $0.kind == .stream }.compactMap(\.text).joined()
        XCTAssertTrue(stdout.contains("sum = 6.0"), stdout)
        XCTAssertEqual(run.last?.status, "ok")

        let tests = try await collect(until: client.runTests(ws.testTarget(for: "scratch.py")))
        let report = try XCTUnwrap(tests.compactMap(\.testReport).first)
        XCTAssertEqual(report.outcome, "passed")
        XCTAssertEqual(report.passed, 1)

        // Lesson flow: open, predict, fail, then pass with the reference solution.
        let opened = try await collect(until: client.openLesson("1.1"))
        let lesson = try XCTUnwrap(opened.compactMap(\.lesson).first)
        XCTAssertEqual(lesson.file, "m01/slope.py")
        XCTAssertNil(lesson.explain)

        let predicted = try await collect(until: client.submitPrediction(lesson: "1.1", value: "3"))
        XCTAssertEqual(predicted.compactMap(\.lesson).first?.hasPrediction, true)

        let failing = try await collect(until: client.checkLesson("1.1"))
        XCTAssertEqual(failing.compactMap(\.testReport).first?.outcome, "failed")

        let solution = try String(
            contentsOf: content.appendingPathComponent("m01-backprop/solutions/slope.py"), encoding: .utf8
        )
        try ws.write("m01/slope.py", contents: solution)
        let passing = try await collect(until: client.checkLesson("1.1"))
        XCTAssertEqual(passing.compactMap(\.testReport).first?.outcome, "passed")
        let passed = try XCTUnwrap(passing.compactMap(\.lesson).first)
        XCTAssertTrue(passed.isPassed)
        XCTAssertEqual(passed.progress.feedback?.correct, true)
        XCTAssertNotNil(passed.explain)
    }
}
