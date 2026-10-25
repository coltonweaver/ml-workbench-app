import XCTest
@testable import WorkbenchCore

final class InstrumentTests: XCTestCase {
    /// L = a*b + a, after backward(), as the Python helper publishes it.
    let graphEvent = """
    {"type":"display_data","id":"r3","display_id":"abc123","data":{
      "text/plain":"<graph: L>",
      "application/vnd.workbench.graph+json":{"kind":"graph","title":"L",
        "nodes":[{"id":0,"op":"","label":"a","data":2.0,"grad":-2.0},
                 {"id":1,"op":"","label":"","data":-3.0,"grad":2.0},
                 {"id":2,"op":"*","label":"","data":-6.0,"grad":1.0},
                 {"id":3,"op":"+","label":"","data":-4.0,"grad":1.0}],
        "edges":[[0,2],[1,2],[2,3],[0,3]],
        "order":[0,1,2,3],"root":3,"has_backward":true}}}
    """

    func graph() throws -> GraphPayload {
        let event = try JSONDecoder().decode(SidecarEvent.self, from: Data(graphEvent.utf8))
        XCTAssertEqual(event.displayID, "abc123")
        guard case .graph(let g)? = Instrument(data: try XCTUnwrap(event.data)) else {
            XCTFail("expected a graph instrument")
            throw XCTSkip()
        }
        return g
    }

    func testDecodesGraphThroughJSONValue() throws {
        let g = try graph()
        XCTAssertEqual(g.nodes.map(\.id), [0, 1, 2, 3])
        XCTAssertEqual(g.nodes[0].label, "a")
        XCTAssertEqual(g.nodes[0].grad, -2.0)
        XCTAssertTrue(g.hasBackward)
        XCTAssertEqual(Instrument.graph(g).title, "L")
    }

    func testGraphLayoutPutsInputsLeft() throws {
        let layout = try graph().layout()
        XCTAssertEqual(layout[0]?.column, 0)
        XCTAssertEqual(layout[1]?.column, 0)
        XCTAssertEqual(layout[2]?.column, 1)
        XCTAssertEqual(layout[3]?.column, 2, "L is one column right of its furthest input")
        XCTAssertEqual(layout[0]?.row, 0)
        XCTAssertEqual(layout[1]?.row, 1)
    }

    func testStepThroughRevealsForwardThenBackward() throws {
        let g = try graph()
        XCTAssertEqual(g.stepCount, 8)
        let start = g.reveal(at: 0)
        XCTAssertTrue(start.values.isEmpty)
        XCTAssertNil(start.current)

        let midForward = g.reveal(at: 3)
        XCTAssertEqual(midForward.values, [0, 1, 2])
        XCTAssertEqual(midForward.current, 2)
        XCTAssertTrue(midForward.grads.isEmpty)

        let firstBackward = g.reveal(at: 5)
        XCTAssertEqual(firstBackward.values.count, 4)
        XCTAssertEqual(firstBackward.grads, [3], "backward starts at the root")
        XCTAssertEqual(firstBackward.current, 3)

        let end = g.reveal(at: 8)
        XCTAssertEqual(end.grads, [0, 1, 2, 3])
    }

    func testLandscape1D() throws {
        let json = """
        {"application/vnd.workbench.landscape+json":{"kind":"landscape","title":null,"dims":1,
          "labels":["w"],"x":[0,1,2,3,4],"line":[16,9,4,1,null],"path":[[0,16],[2,4]]}}
        """
        let data = try JSONDecoder().decode([String: JSONValue].self, from: Data(json.utf8))
        guard case .landscape(let l)? = Instrument(data: data) else { return XCTFail("expected a landscape") }
        XCTAssertEqual(l.dims, 1)
        XCTAssertEqual(l.value(atX: 1.5) ?? .nan, 6.5, accuracy: 1e-9)
        XCTAssertNil(l.value(atX: 3.5), "no value next to a null sample")
        XCTAssertNil(l.value(atX: 9), "nothing outside the range")
        XCTAssertEqual(l.valueRange, 1...16)
        XCTAssertEqual(l.gradient(atX: 1.5)?.first ?? .nan, -5, accuracy: 1e-9)
        XCTAssertEqual(l.path[1], [2, 4])
    }

    func testLandscape2DBilinear() throws {
        // f(x, y) = x + 10 y on a 3 x 3 grid over [0, 2] x [0, 2].
        let l = LandscapePayload(
            title: "plane", dims: 2, labels: ["x", "y"], x: [0, 1, 2], y: [0, 1, 2], line: nil,
            grid: [[0, 1, 2], [10, 11, 12], [20, 21, 22]], path: []
        )
        XCTAssertEqual(l.value(atX: 0.5, y: 1.5) ?? .nan, 15.5, accuracy: 1e-9)
        let g = try XCTUnwrap(l.gradient(atX: 1, y: 1))
        XCTAssertEqual(g[0], 1, accuracy: 1e-9)
        XCTAssertEqual(g[1], 10, accuracy: 1e-9)
    }

    func testPlainDisplayIsNotAnInstrument() {
        XCTAssertNil(Instrument(data: ["text/plain": .string("42")]))
    }
}
