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

final class ImageInstrumentTests: XCTestCase {
    func testImagesPayload() throws {
        let pixels = Data([0, 128, 255, 64]).base64EncodedString()
        let json = """
        {"application/vnd.workbench.images+json": {"kind": "images", "title": "filters", "explain": [],
          "images": [{"label": "c0", "width": 2, "height": 2, "channels": 1, "pixels": "\(pixels)", "range": [-1, 1]},
                     {"label": null, "width": 3, "height": 2, "channels": 1, "pixels": "\(pixels)"}]}}
        """
        let data = try JSONDecoder().decode([String: JSONValue].self, from: Data(json.utf8))
        guard case .images(let p)? = Instrument(data: data) else { return XCTFail("not decoded") }
        XCTAssertEqual(p.images[0].bytes, Data([0, 128, 255, 64]))
        XCTAssertNil(p.images[1].bytes, "4 bytes can't be a 3 × 2 image")
        XCTAssertEqual(p.images[0].range, [-1, 1])
    }
}

final class InterpInstrumentTests: XCTestCase {
    private func decode(_ json: String) throws -> Instrument? {
        Instrument(data: try JSONDecoder().decode([String: JSONValue].self, from: Data(json.utf8)))
    }

    func testFeaturesPayload() throws {
        let json = """
        {"application/vnd.workbench.features+json": {"kind": "features", "title": "SAE", "explain": [],
          "features": [{"id": 3, "label": null, "density": 0.25, "max_act": 2.0,
                        "examples": [{"tokens": ["g", "h"], "acts": [0.5, 2.0], "highlight": 1}],
                        "up": [{"token": "h", "value": 1.5}], "down": []}]}}
        """
        guard case .features(let p)? = try decode(json) else { return XCTFail("not decoded") }
        XCTAssertEqual(p.features[0].maxAct, 2.0)
        XCTAssertEqual(p.features[0].name, "#3")
        XCTAssertEqual(p.features[0].examples[0].highlight, 1)
    }

    func testAttributionPayload() throws {
        let json = """
        {"application/vnd.workbench.attribution+json": {"kind": "attribution", "title": null, "explain": [],
          "tokens": ["a", "b"], "components": ["emb", "a0"], "values": [[1, -3], [2, null]],
          "target": {"pos": 1, "token": "c"},
          "graph": {"nodes": [{"component": 0, "pos": 0, "effect": 1}, {"component": 1, "pos": 1, "effect": -2}],
                    "edges": [{"source": 0, "target": 1, "weight": 0.5}]}}}
        """
        guard case .attribution(let p)? = try decode(json) else { return XCTFail("not decoded") }
        XCTAssertEqual(p.maxMagnitude, 3)
        XCTAssertEqual(p.totals, [3, -3])
        XCTAssertEqual(p.graph?.edges.first?.target, 1)
        XCTAssertEqual(signedFraction(-6, magnitude: 3), -1)
        XCTAssertEqual(Instrument.attribution(p).title, "Logit attribution")
    }
}

final class ChartInstrumentTests: XCTestCase {
    private func decode(_ json: String) throws -> Instrument? {
        Instrument(data: try JSONDecoder().decode([String: JSONValue].self, from: Data(json.utf8)))
    }

    func testPlotPayload() throws {
        let json = """
        {"application/vnd.workbench.plot+json": {"kind": "plot", "title": "scaling", "x_label": "N", "y_label": null,
          "log_x": true, "log_y": false, "explain": [],
          "series": [{"name": "data", "kind": "scatter", "x": [0, 10, 100], "y": [3, null, 2]}]}}
        """
        guard case .plot(let p)? = try decode(json) else { return XCTFail("not decoded") }
        XCTAssertTrue(p.logX)
        XCTAssertEqual(p.series[0].points(logX: true), [PlotSeries.Point(x: 100, y: 2)],
                       "x = 0 can't go on a log axis and a null y is skipped")
        XCTAssertEqual(p.series[0].points().count, 2)
    }

    func testTimelinePayload() throws {
        let json = """
        {"application/vnd.workbench.timeline+json": {"kind": "timeline", "title": null, "unit": "ms", "explain": [],
          "lanes": ["a", "b"], "metrics": {"bubble": 0.25},
          "spans": [{"lane": 0, "start": 0, "end": 2, "label": "F0", "kind": "forward"},
                    {"lane": 1, "start": 2, "end": 4, "label": null, "kind": "backward"},
                    {"lane": 1, "start": 0, "end": 1, "label": null, "kind": "forward"}],
          "below": []}}
        """
        guard case .timeline(let t)? = try decode(json) else { return XCTFail("not decoded") }
        XCTAssertEqual(t.timeRange, 0...4)
        XCTAssertEqual(t.kinds, ["forward", "backward"])
        XCTAssertEqual(t.utilization, 5.0 / 8.0, accuracy: 1e-12)
        XCTAssertEqual(t.metrics["bubble"], 0.25)
    }
}
