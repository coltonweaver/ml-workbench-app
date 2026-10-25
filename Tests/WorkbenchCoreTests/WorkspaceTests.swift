import XCTest
@testable import WorkbenchCore

final class WorkspaceTests: XCTestCase {
    var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("ws-\(UUID().uuidString)")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testPrepareSeedsEmptyWorkspaceAndInitsGit() throws {
        let ws = Workspace(root: root)
        try ws.prepare()
        XCTAssertEqual(try ws.pythonFiles(), ["scratch.py", "tests/test_scratch.py"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent(".git").path))
    }

    func testPrepareLeavesExistingWorkspaceAlone() throws {
        let ws = Workspace(root: root)
        try ws.write("mine.py", contents: "x = 1\n")
        try ws.prepare(initGit: false)
        XCTAssertEqual(try ws.pythonFiles(), ["mine.py"])
        XCTAssertEqual(try ws.read("mine.py"), "x = 1\n")
    }

    func testPythonFilesSkipsCachesAndVenvs() throws {
        let ws = Workspace(root: root)
        try ws.write("m01/engine.py", contents: "")
        try ws.write("m01/__pycache__/engine.cpython-312.py", contents: "")
        try ws.write(".venv/lib/site.py", contents: "")
        try ws.write("notes.md", contents: "")
        XCTAssertEqual(try ws.pythonFiles(), ["m01/engine.py"])
    }

    func testRejectsPathsOutsideWorkspace() throws {
        let ws = Workspace(root: root)
        XCTAssertThrowsError(try ws.write("../escape.py", contents: ""))
    }

    func testTestTargetMapping() throws {
        let ws = Workspace(root: root)
        try ws.write("m01/engine.py", contents: "")
        try ws.write("tests/test_engine.py", contents: "")
        try ws.write("m01/value.py", contents: "")
        try ws.write("m01/test_value.py", contents: "")
        try ws.write("m01/other.py", contents: "")

        XCTAssertEqual(ws.testTarget(for: "m01/engine.py"), "tests/test_engine.py")
        XCTAssertEqual(ws.testTarget(for: "m01/value.py"), "m01/test_value.py")
        XCTAssertEqual(ws.testTarget(for: "m01/other.py"), "tests")
        XCTAssertEqual(ws.testTarget(for: "tests/test_engine.py"), "tests/test_engine.py")
    }
}
