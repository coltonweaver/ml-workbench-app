import Security
import XCTest
@testable import WorkbenchCore

final class TutorTests: XCTestCase {
    func testAskTutorRequestEncoding() throws {
        let request = SidecarRequest(
            id: "r9", op: .askTutor, lesson: "1.4", question: "why?",
            history: [TutorTurn(role: .user, text: "q"), TutorTurn(role: .assistant, text: "a")],
            apiKey: "sk-test", tensor: .object(["name": .string("w"), "shape": .array([.integer(3)])])
        )
        let text = String(decoding: try request.encodedLine(), as: UTF8.self)
        XCTAssertEqual(text, #"{"api_key":"sk-test","history":[{"role":"user","text":"q"},{"role":"assistant","text":"a"}],"id":"r9","lesson":"1.4","op":"ask_tutor","question":"why?","tensor":{"name":"w","shape":[3]}}"# + "\n")
    }

    func testDecodesTutorEvents() throws {
        let delta = try JSONDecoder().decode(SidecarEvent.self, from: Data(#"{"type":"tutor_delta","id":"r1","text":"Hi"}"#.utf8))
        XCTAssertEqual(delta.kind, .tutorDelta)
        XCTAssertEqual(delta.text, "Hi")
        let done = try JSONDecoder().decode(SidecarEvent.self, from: Data(#"{"type":"tutor_done","id":"r1","stop_reason":"refusal","note":"declined"}"#.utf8))
        XCTAssertEqual(done.kind, .tutorDone)
        XCTAssertEqual(done.stopReason, "refusal")
        XCTAssertEqual(done.note, "declined")
    }

    func testChatStreamsAndBuildsHistory() {
        var chat = TutorChat()
        XCTAssertEqual(chat.history, [])
        chat.ask("first", request: "r1")
        XCTAssertTrue(chat.isAnswering)
        chat.receive(delta: "Look ", request: "r1")
        chat.receive(delta: "here.", request: "r1")
        chat.receive(delta: "ignored", request: "other")
        chat.finish(request: "r1", stopReason: "end_turn", note: nil)
        chat.end(request: "r1")
        XCTAssertFalse(chat.isAnswering)
        XCTAssertEqual(chat.history, [TutorTurn(role: .user, text: "first"), TutorTurn(role: .assistant, text: "Look here.")])
    }

    func testStoppedAndFailedAnswersArentReplayed() {
        var chat = TutorChat()
        chat.ask("one", request: "r1")
        chat.receive(delta: "partial", request: "r1")
        chat.finish(request: "r1", stopReason: "cancelled", note: nil)
        chat.end(request: "r1")
        XCTAssertEqual(chat.messages.last?.note, "Stopped.")
        XCTAssertEqual(chat.history, [TutorTurn(role: .user, text: "one")])

        chat.ask("two", request: "r2")
        chat.fail(request: "r2", message: "Add your Claude API key in Settings to use the tutor.")
        chat.end(request: "r2")
        XCTAssertTrue(chat.messages.last?.isError == true)
        XCTAssertEqual(chat.history, [TutorTurn(role: .user, text: "one")])

        chat.ask("three", request: "r3")
        chat.end(request: "r3")  // done with no text at all
        XCTAssertEqual(chat.messages.last?.note, "No answer.")
        chat.clear()
        XCTAssertTrue(chat.messages.isEmpty)
    }

    func testExplainSummariesComeFromAnyWorkbenchPayload() throws {
        let json = #"{"application/vnd.workbench.graph+json":{"kind":"graph","explain":[{"name":"node values","shape":[3]}]},"text/plain":"<graph>"}"#
        let data = try JSONDecoder().decode([String: JSONValue].self, from: Data(json.utf8))
        let summaries = Instrument.explainSummaries(data: data)
        XCTAssertEqual(summaries.count, 1)
        XCTAssertEqual(summaries.first?.name, "node values")
        XCTAssertEqual(Instrument.explainSummaries(data: ["text/plain": .string("x")]), [])
    }

    func testKeychainRoundTrip() throws {
        let item = KeychainItem(service: "ML Workbench Tests", account: UUID().uuidString)
        do {
            XCTAssertNil(try item.read())
            try item.save("one")
            XCTAssertEqual(try item.read(), "one")
            try item.save("two")
            XCTAssertEqual(try item.read(), "two")
            try item.delete()
            XCTAssertNil(try item.read())
        } catch let failure as KeychainItem.Failure where failure.status == errSecInteractionNotAllowed
                    || failure.status == errSecNotAvailable || failure.status == -34018 {
            throw XCTSkip("no usable keychain here: \(failure.localizedDescription)")
        }
    }
}
