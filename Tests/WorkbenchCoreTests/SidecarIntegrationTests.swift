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

        let client = SidecarClient(launch: SidecarLaunch(uv: uv, pythonProject: project, workspace: root))
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
    }
}
