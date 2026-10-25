import XCTest
@testable import WorkbenchCore

final class LessonProtocolTests: XCTestCase {
    func testLessonRequestsEncode() throws {
        let line = try SidecarRequest(id: "r5", op: .submitPrediction, lesson: "1.2", value: "2").encodedLine()
        XCTAssertEqual(String(decoding: line, as: UTF8.self),
                       #"{"id":"r5","lesson":"1.2","op":"submit_prediction","value":"2"}"# + "\n")
        let check = try SidecarRequest(id: "r6", op: .checkLesson, lesson: "1.1").encodedLine()
        XCTAssertEqual(String(decoding: check, as: UTF8.self), #"{"id":"r6","lesson":"1.1","op":"check_lesson"}"# + "\n")
    }

    func testDecodesLessonsEvent() throws {
        let json = """
        {"type":"lessons","id":"r1","modules":[{"id":"m01","number":1,"title":"Gradient descent and backprop",
         "summary":"From slopes to backprop.","lessons":[{"id":"1.1","title":"Which way is downhill?","status":"passed"},
         {"id":"1.2","title":"Walking downhill","status":"new"}]}]}
        """
        let event = try JSONDecoder().decode(SidecarEvent.self, from: Data(json.utf8))
        XCTAssertEqual(event.kind, .lessons)
        let module = try XCTUnwrap(event.modules?.first)
        XCTAssertEqual(module.lessons.map(\.status), ["passed", "new"])
    }

    func testDecodesLessonBeforeAndAfterPassing() throws {
        let before = """
        {"type":"lesson","id":"r2","lesson":{"id":"1.2","title":"Walking downhill","module_id":"m01",
         "module_title":"Gradient descent and backprop","file":"m01/descent.py","concepts":["gradient-descent"],
         "hook":"h","build":"b","explain":null,"stretch":null,
         "predict":{"kind":"choice","prompt":"p","options":["a","b","c"]},
         "progress":{"status":"in_progress","attempts":0,"prediction":null,"feedback":null}}}
        """
        let lesson = try XCTUnwrap(JSONDecoder().decode(SidecarEvent.self, from: Data(before.utf8)).lesson)
        XCTAssertEqual(lesson.moduleID, "m01")
        XCTAssertFalse(lesson.hasPrediction)
        XCTAssertFalse(lesson.isPassed)
        XCTAssertNil(lesson.explain)
        XCTAssertEqual(lesson.predict.options, ["a", "b", "c"])

        let after = """
        {"type":"lesson","lesson":{"id":"1.1","title":"t","module_id":"m01","module_title":"m",
         "file":"m01/slope.py","concepts":[],"hook":"h","build":"b","explain":"e","stretch":"s",
         "predict":{"kind":"text","prompt":"p","options":null},
         "progress":{"status":"passed","attempts":2,
           "prediction":{"value":"it diverges","submitted_at":"2026-10-25T21:00:00+0000"},
           "feedback":{"correct":null,"yours":"it diverges","answer":null}}}}
        """
        let passed = try XCTUnwrap(JSONDecoder().decode(SidecarEvent.self, from: Data(after.utf8)).lesson)
        XCTAssertTrue(passed.isPassed)
        XCTAssertEqual(passed.progress.prediction?.value, "it diverges")
        XCTAssertNil(passed.progress.feedback?.correct)
        XCTAssertEqual(passed.explain, "e")
    }

    func testFindsContentInSourceTree() {
        let content = SidecarLaunch.findContent(environment: [:])
        XCTAssertNotNil(content)
        XCTAssertTrue(FileManager.default.fileExists(atPath: content!.appendingPathComponent("m01-backprop/module.yaml").path))
    }

    func testLaunchPassesContent() {
        let launch = SidecarLaunch(
            uv: URL(fileURLWithPath: "/bin/uv"), pythonProject: URL(fileURLWithPath: "/p"),
            workspace: URL(fileURLWithPath: "/w"), content: URL(fileURLWithPath: "/c")
        )
        XCTAssertEqual(Array(launch.arguments.suffix(2)), ["--content", "/c"])
    }
}

final class MarkdownBlockTests: XCTestCase {
    func testParagraphsJoinLinesAndSplitOnBlankLines() {
        let blocks = MarkdownBlock.parse("one\ntwo\n\nthree")
        XCTAssertEqual(blocks, [.paragraph("one two"), .paragraph("three")])
    }

    func testCodeFencesKeepLinesVerbatim() {
        let blocks = MarkdownBlock.parse("Try:\n\n```python\nx = 1\n  y = 2\n```\nafter")
        XCTAssertEqual(blocks, [.paragraph("Try:"), .code("x = 1\n  y = 2"), .paragraph("after")])
    }

    func testListsAndContinuations() {
        let blocks = MarkdownBlock.parse("- a\n- b continues\n  on the next line\n\n1. first\n2. second")
        XCTAssertEqual(blocks, [
            .list(["a", "b continues on the next line"], ordered: false),
            .list(["first", "second"], ordered: true),
        ])
    }

    func testTablesDropTheSeparatorRow() {
        let blocks = MarkdownBlock.parse("| w | cost |\n|---|---|\n| 10 | 50.00 |")
        XCTAssertEqual(blocks, [.table([["w", "cost"], ["10", "50.00"]])])
    }

    func testHeadings() {
        XCTAssertEqual(MarkdownBlock.parse("### Why\ntext"), [.heading("Why"), .paragraph("text")])
    }
}
