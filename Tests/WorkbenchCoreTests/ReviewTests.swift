import XCTest
@testable import WorkbenchCore

final class ReviewTests: XCTestCase {
    func testReviewRequestsEncode() throws {
        let predict = try SidecarRequest(id: "r1", op: .submitPrediction, lesson: "1.1", value: "3",
                                         confidence: 0.7).encodedLine()
        XCTAssertEqual(String(decoding: predict, as: UTF8.self),
                       #"{"confidence":0.7,"id":"r1","lesson":"1.1","op":"submit_prediction","value":"3"}"# + "\n")
        let get = try SidecarRequest(id: "r2", op: .getReview).encodedLine()
        XCTAssertEqual(String(decoding: get, as: UTF8.self), #"{"id":"r2","op":"get_review"}"# + "\n")
        let answer = try SidecarRequest(id: "r3", op: .answerWarmup, value: "6",
                                        warmup: "derivative|1.1|0").encodedLine()
        XCTAssertEqual(String(decoding: answer, as: UTF8.self),
                       #"{"id":"r3","op":"answer_warmup","value":"6","warmup":"derivative|1.1|0"}"# + "\n")
    }

    func testDecodesReviewEvent() throws {
        // As the sidecar sends it (see python/tests/test_review.py).
        let json = """
        {"type":"review","id":"r4","review":{
         "warmups":[{"id":"derivative|1.1|p","concept":"derivative","lesson":"1.1",
           "lesson_title":"Which way is downhill?","context":"You run a service…","kind":"number",
           "prompt":"About how much does the cost change?","options":null,"mistakes":2,
           "last_message":"off by 2x"}],
         "score":{"count":4,"correct":3,"hit_rate":0.75,"rated":3,"brier":0.3566,
           "overconfidence":0.1,"buckets":[{"confidence":0.5,"count":1,"hit_rate":1.0},
           {"confidence":0.9,"count":2,"hit_rate":0.5}],
           "history":[{"lesson":"1.1","correct":true,"confidence":0.9,"hit_rate":1.0},
           {"lesson":"1.2","correct":false,"confidence":null,"hit_rate":0.5}]},
         "result":{"id":"kv-cache|7.2|0","concept":"kv-cache","correct":false,"yours":"1","answer":"0.5",
           "explain":"2 × 32 × 4096 × 2 bytes","lesson":"7.2","lesson_title":"The KV cache"}}}
        """
        let event = try JSONDecoder().decode(SidecarEvent.self, from: Data(json.utf8))
        XCTAssertEqual(event.kind, .review)
        let review = try XCTUnwrap(event.review)
        let warmup = try XCTUnwrap(review.warmups.first)
        XCTAssertEqual(warmup.lessonTitle, "Which way is downhill?")
        XCTAssertEqual(warmup.mistakes, 2)
        XCTAssertEqual(warmup.lastMessage, "off by 2x")
        XCTAssertNil(warmup.options)
        XCTAssertEqual(review.score.hitRate, 0.75)
        XCTAssertEqual(review.score.buckets.map(\.hitRate), [1.0, 0.5])
        XCTAssertEqual(review.score.history.map(\.confidence), [0.9, nil])
        XCTAssertEqual(review.result?.correct, false)
        XCTAssertEqual(review.result?.lessonTitle, "The KV cache")
    }

    func testDecodesEmptyReview() throws {
        let json = """
        {"type":"review","id":"r5","review":{"warmups":[],"score":{"count":0,"correct":0,"hit_rate":null,
         "rated":0,"brier":null,"overconfidence":null,"buckets":[],"history":[]},"result":null}}
        """
        let review = try XCTUnwrap(JSONDecoder().decode(SidecarEvent.self, from: Data(json.utf8)).review)
        XCTAssertEqual(review.score, .empty)
        XCTAssertTrue(review.warmups.isEmpty)
        XCTAssertNil(review.result)
    }

    func testConfidenceOnPrediction() throws {
        let json = #"{"value":"3","submitted_at":"2026-10-25T10:00:00+0000","confidence":0.9}"#
        let prediction = try JSONDecoder().decode(LessonDetail.Prediction.self, from: Data(json.utf8))
        XCTAssertEqual(prediction.confidence, 0.9)
        let old = #"{"value":"3","submitted_at":"2026-10-25T10:00:00+0000"}"#
        XCTAssertNil(try JSONDecoder().decode(LessonDetail.Prediction.self, from: Data(old.utf8)).confidence)
    }

    func testVerdict() {
        XCTAssertTrue(PredictionScore.empty.verdict.hasPrefix("Your score starts"))
        var score = PredictionScore(count: 4, correct: 3, hitRate: 0.75, rated: 2, brier: 0.1,
                                    overconfidence: 0.3, buckets: [], history: [])
        XCTAssertTrue(score.verdict.hasPrefix("Right 3 of 4."), "too few rated to judge calibration")
        score.rated = 6
        XCTAssertEqual(score.verdict, "Overconfident: your confidence runs 30 points above how often you're right.")
        score.overconfidence = -0.2
        XCTAssertTrue(score.verdict.hasPrefix("Underconfident: you're right 20 points"))
        score.overconfidence = 0.04
        XCTAssertTrue(score.verdict.hasPrefix("Well calibrated"))
    }

    func testConceptName() {
        let warmup = Warmup(id: "kv-cache|7.2|0", concept: "kv-cache", lesson: "7.2", lessonTitle: "The KV cache",
                            context: nil, kind: "number", prompt: "p", options: nil, mistakes: 2, lastMessage: nil)
        XCTAssertEqual(warmup.conceptName, "KV cache")
        var other = warmup
        other.concept = "chain-rule"
        XCTAssertEqual(other.conceptName, "Chain rule")
    }
}
