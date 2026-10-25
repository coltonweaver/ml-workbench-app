import Foundation

/// The learner's workspace: a plain git repo of .py files on disk. The app
/// reads and writes these files directly; the learner can also edit them in
/// any other editor.
public struct Workspace: Sendable {
    public let root: URL

    public init(root: URL) {
        self.root = root.standardizedFileURL
    }

    /// `WORKBENCH_WORKSPACE`, else `~/ml-workbench`.
    public static func defaultRoot(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        if let explicit = environment["WORKBENCH_WORKSPACE"] {
            return URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("ml-workbench")
    }

    static let starterFiles: [String: String] = [
        "scratch.py": """
        # Scratch space. Press Run (⌘R) to execute this file in the kernel;
        # its variables stay available afterwards.
        import numpy as np

        w = np.array([1.0, 2.0, 3.0])
        print("w =", w, "sum =", w.sum())

        """,
        "tests/test_scratch.py": """
        # Press Test (⌘U) with scratch.py open to run this.
        import numpy as np


        def test_numpy_works():
            assert np.allclose(np.array([1.0, 2.0]) * 2, [2.0, 4.0])

        """,
        ".gitignore": """
        __pycache__/
        .pytest_cache/
        .venv/

        """,
    ]

    /// Creates the folder, starter files and git repo if they aren't there yet.
    /// Never overwrites an existing file.
    public func prepare(initGit: Bool = true) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        let isEmpty = try fm.contentsOfDirectory(atPath: root.path).filter { $0 != ".DS_Store" }.isEmpty
        if isEmpty {
            for (path, contents) in Self.starterFiles {
                let url = root.appendingPathComponent(path)
                try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try contents.write(to: url, atomically: true, encoding: .utf8)
            }
        }
        if initGit, !fm.fileExists(atPath: root.appendingPathComponent(".git").path) {
            try runGit(["init", "--quiet"])
        }
    }

    /// Workspace-relative paths of every .py file, sorted, skipping hidden
    /// folders, virtualenvs and caches.
    public func pythonFiles() throws -> [String] {
        let skip: Set<String> = ["__pycache__", ".venv", "venv", "node_modules"]
        guard let walker = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }
        var files: [String] = []
        for case let url as URL in walker {
            if skip.contains(url.lastPathComponent) {
                walker.skipDescendants()
                continue
            }
            if url.pathExtension == "py", let rel = relativePath(of: url) {
                files.append(rel)
            }
        }
        return files.sorted()
    }

    public func read(_ path: String) throws -> String {
        try String(contentsOf: try url(for: path), encoding: .utf8)
    }

    public func write(_ path: String, contents: String) throws {
        let url = try url(for: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Which tests the Test button runs for the open file:
    /// a test file runs itself; `pkg/foo.py` runs `tests/test_foo.py` or
    /// `pkg/test_foo.py` if either exists; otherwise the whole `tests/` folder.
    public func testTarget(for path: String) -> String {
        let name = (path as NSString).lastPathComponent
        if name.hasPrefix("test_") || name.hasSuffix("_test.py") {
            return path
        }
        let dir = (path as NSString).deletingLastPathComponent
        let testName = "test_" + name
        let candidates = [
            ("tests" as NSString).appendingPathComponent(testName),
            ((dir as NSString).appendingPathComponent("tests") as NSString).appendingPathComponent(testName),
            (dir as NSString).appendingPathComponent(testName),
        ]
        let fm = FileManager.default
        if let hit = candidates.first(where: { fm.fileExists(atPath: root.appendingPathComponent($0).path) }) {
            return hit
        }
        return "tests"
    }

    func url(for path: String) throws -> URL {
        let url = root.appendingPathComponent(path).standardizedFileURL
        guard relativePath(of: url) != nil else {
            throw CocoaError(.fileReadInvalidFileName, userInfo: [NSFilePathErrorKey: path])
        }
        return url
    }

    func relativePath(of url: URL) -> String? {
        let base = root.resolvingSymlinksInPath().path
        let full = url.standardizedFileURL.resolvingSymlinksInPath().path
        guard full.hasPrefix(base + "/") else { return nil }
        return String(full.dropFirst(base.count + 1))
    }

    private func runGit(_ args: [String]) throws {
        let git = Process()
        git.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        git.arguments = args
        git.currentDirectoryURL = root
        git.standardOutput = FileHandle.nullDevice
        git.standardError = FileHandle.nullDevice
        try git.run()
        git.waitUntilExit()
    }
}
