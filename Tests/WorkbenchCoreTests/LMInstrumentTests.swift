import XCTest
@testable import WorkbenchCore

final class LMInstrumentTests: XCTestCase {
    private func instrument(_ json: String) throws -> Instrument? {
        let data = try JSONDecoder().decode([String: JSONValue].self, from: Data(json.utf8))
        return Instrument(data: data)
    }

    func testTrainingPayload() throws {
        let json = """
        {"application/vnd.workbench.training+json": {"kind": "training", "title": "mlp", "stride": 2, "done": false,
          "explain": [],
          "points": [
            {"step": 0, "loss": 3.0, "lr": 0.1, "grad_norms": {"W1": 2.0, "C": 0.5}},
            {"step": 2, "loss": 2.0, "val_loss": 2.5, "tokens_per_sec": 1000.0, "grad_norms": {"W2": 1.0}},
            {"step": 4, "loss": 1.0, "val_loss": 2.2, "tokens_per_sec": 3000.0, "metrics": {"acc": 0.5}}
          ]}}
        """
        guard case .training(let t)? = try instrument(json) else { return XCTFail("not decoded") }
        XCTAssertEqual(t.title, "mlp")
        XCTAssertEqual(t.layers, ["C", "W1", "W2"])
        XCTAssertEqual(t.metricNames, ["acc"])
        XCTAssertEqual(t.latestLoss, 1.0)
        XCTAssertEqual(t.bestValLoss?.step, 4)
        XCTAssertEqual(t.recentTokensPerSec, 2000.0)
        let smooth = t.smoothedLoss(weight: 0.5)
        XCTAssertEqual(smooth.map(\.loss), [3.0, 2.5, 1.75])
    }

    func testEmbeddingPayload() throws {
        let json = """
        {"application/vnd.workbench.embedding+json": {"kind": "embedding", "title": null, "method": "pca", "dim": 10,
          "points": [[0.0, 0.0], [1.0, 1.0], [null, 2.0]], "labels": [".", "a", "b"],
          "classes": ["special", "vowel", "consonant"], "explained_variance": [0.6, 0.3], "explain": []}}
        """
        guard case .embedding(let e)? = try instrument(json) else { return XCTFail("not decoded") }
        XCTAssertEqual(e.plotted.count, 2, "points with a null coordinate aren't plotted")
        XCTAssertEqual(e.nearest(x: 0.9, y: 0.8)?.label, "a")
        XCTAssertEqual(e.explainedVariance, [0.6, 0.3])
    }

    func testAttentionPayloadAndRerunCode() throws {
        let json = """
        {"application/vnd.workbench.attention+json": {"kind": "attention", "title": "gpt", "prompt": "hi",
          "tokens": ["h", "i"], "callback": "attention-3", "explain": [],
          "layers": [{"name": "layer 0", "heads": [[[1.0, 0.0], [0.4, 0.6]]]}]}}
        """
        guard case .attention(let a)? = try instrument(json) else { return XCTFail("not decoded") }
        XCTAssertEqual(a.layers[0].heads[0][1][1], 0.6)
        XCTAssertEqual(a.rerunCode(prompt: #"say "hi"\#n"#),
                       #"import workbench as _wb; _wb.rerun("attention-3", "say \"hi\"\n")"#)
        var noCallback = a
        noCallback.callback = nil
        XCTAssertNil(noCallback.rerunCode(prompt: "x"))
    }

    func testMatrixPayloadAndHeatScale() throws {
        let json = """
        {"application/vnd.workbench.matrix+json": {"kind": "matrix", "title": "counts", "values": [[0, 9], [3, null]],
          "rows": ["a", "b"], "cols": ["a", "b"], "log_scale": true, "explain": []}}
        """
        guard case .matrix(let m)? = try instrument(json) else { return XCTFail("not decoded") }
        XCTAssertEqual(m.valueRange, 0...9)
        XCTAssertTrue(m.logScale)
        XCTAssertEqual(heatFraction(9, in: 0...9, log: true), 1, accuracy: 1e-12)
        XCTAssertEqual(heatFraction(4.5, in: 0...9, log: false), 0.5, accuracy: 1e-12)
        XCTAssertGreaterThan(heatFraction(1, in: 0...9, log: true), 1.0 / 9, "log scale lifts small values")
    }
}
