import Foundation

/// A visual widget's data, as published by the `workbench` Python helper.
public enum Instrument: Equatable, Sendable {
    case graph(GraphPayload)
    case landscape(LandscapePayload)

    public static let graphMIME = "application/vnd.workbench.graph+json"
    public static let landscapeMIME = "application/vnd.workbench.landscape+json"

    /// Picks the instrument out of a display's MIME bundle, if it has one.
    public init?(data: [String: JSONValue]) {
        if let value = data[Self.graphMIME], let payload = try? value.decode(GraphPayload.self) {
            self = .graph(payload)
        } else if let value = data[Self.landscapeMIME], let payload = try? value.decode(LandscapePayload.self) {
            self = .landscape(payload)
        } else {
            return nil
        }
    }

    public var title: String {
        switch self {
        case .graph(let g): g.title ?? "Computation graph"
        case .landscape(let l): l.title ?? "Loss landscape"
        }
    }
}

extension JSONValue {
    /// Re-decodes this tree as a typed value.
    public func decode<T: Decodable>(_ type: T.Type) throws -> T {
        try JSONDecoder().decode(T.self, from: JSONEncoder().encode(self))
    }
}

// MARK: Computation graph

public struct GraphPayload: Codable, Equatable, Sendable {
    public struct Node: Codable, Equatable, Sendable, Identifiable {
        public var id: Int
        public var op: String
        public var label: String
        public var data: Double?
        public var grad: Double?
    }

    public var title: String?
    public var nodes: [Node]
    /// [child, parent] pairs: the child is an input to the parent.
    public var edges: [[Int]]
    /// Node ids in topological order, inputs first.
    public var order: [Int]
    public var root: Int
    public var hasBackward: Bool

    enum CodingKeys: String, CodingKey {
        case title, nodes, edges, order, root
        case hasBackward = "has_backward"
    }

    /// Column and row for each node: inputs on the left (column 0), each node
    /// one column right of its furthest input, rows in topological order.
    public func layout() -> [Int: (column: Int, row: Int)] {
        var inputs: [Int: [Int]] = [:]
        for edge in edges where edge.count == 2 {
            inputs[edge[1], default: []].append(edge[0])
        }
        var column: [Int: Int] = [:]
        for id in order {
            column[id] = (inputs[id] ?? []).map { (column[$0] ?? 0) + 1 }.max() ?? 0
        }
        var nextRow: [Int: Int] = [:]
        var result: [Int: (column: Int, row: Int)] = [:]
        for id in order {
            let c = column[id] ?? 0
            result[id] = (c, nextRow[c, default: 0])
            nextRow[c, default: 0] += 1
        }
        return result
    }

    /// The step-through has a forward pass (one step per node, inputs first)
    /// and, when backward() was called, a backward pass in reverse.
    public var stepCount: Int { order.count * (hasBackward ? 2 : 1) }

    /// What's revealed after `step` steps (0 = nothing yet).
    public func reveal(at step: Int) -> (values: Set<Int>, grads: Set<Int>, current: Int?) {
        let n = order.count
        let forward = min(step, n)
        let values = Set(order.prefix(forward))
        guard hasBackward, step > n else {
            return (values, [], step > 0 ? order[forward - 1] : nil)
        }
        let backward = min(step - n, n)
        let reversed = Array(order.reversed())
        // A node's grad is final once every node that uses it has run, which
        // in reverse topological order is by the time it's reached.
        return (values, Set(reversed.prefix(backward)), reversed[backward - 1])
    }
}

// MARK: Loss landscape

public struct LandscapePayload: Codable, Equatable, Sendable {
    public var title: String?
    public var dims: Int
    public var labels: [String]
    public var x: [Double]
    /// 2D only.
    public var y: [Double]?
    /// 1D: f at each x. Null where f wasn't finite.
    public var line: [Double?]?
    /// 2D: grid[j][i] = f(x[i], y[j]).
    public var grid: [[Double?]]?
    /// 1D: [x, f]. 2D: [x, y, f]. Entries may be null if they overflowed.
    public var path: [[Double?]]

    /// The range of finite values, for colour and axis scaling.
    public var valueRange: ClosedRange<Double>? {
        let values = (line ?? []).compactMap { $0 } + (grid ?? []).flatMap { $0.compactMap { $0 } }
        guard let lo = values.min(), let hi = values.max() else { return nil }
        return lo...max(hi, lo + 1e-12)
    }

    /// f at x (1D), linearly interpolated between samples.
    public func value(atX px: Double) -> Double? {
        guard let line, let cell = Self.locate(px, in: x) else { return nil }
        let (i, t) = cell
        guard let a = line[i], let b = line[min(i + 1, line.count - 1)] else { return nil }
        return a + (b - a) * t
    }

    /// f at (x, y) (2D), bilinearly interpolated.
    public func value(atX px: Double, y py: Double) -> Double? {
        guard let grid, let y, let cx = Self.locate(px, in: x), let cy = Self.locate(py, in: y) else {
            return nil
        }
        let (i, tx) = cx
        let (j, ty) = cy
        let i1 = min(i + 1, x.count - 1), j1 = min(j + 1, y.count - 1)
        guard let v00 = grid[j][i], let v10 = grid[j][i1], let v01 = grid[j1][i], let v11 = grid[j1][i1] else {
            return nil
        }
        let bottom = v00 + (v10 - v00) * tx
        let top = v01 + (v11 - v01) * tx
        return bottom + (top - bottom) * ty
    }

    /// Slope estimate from the samples: df/dx in 1D, (∂f/∂x, ∂f/∂y) in 2D.
    public func gradient(atX px: Double, y py: Double? = nil) -> [Double]? {
        let hx = Self.step(x)
        if dims == 1 {
            guard let a = value(atX: px - hx), let b = value(atX: px + hx) else { return nil }
            return [(b - a) / (2 * hx)]
        }
        guard let py, let y else { return nil }
        let hy = Self.step(y)
        guard let xa = value(atX: px - hx, y: py), let xb = value(atX: px + hx, y: py),
              let ya = value(atX: px, y: py - hy), let yb = value(atX: px, y: py + hy) else { return nil }
        return [(xb - xa) / (2 * hx), (yb - ya) / (2 * hy)]
    }

    /// Index of the cell containing v and the fraction across it; nil outside.
    static func locate(_ v: Double, in axis: [Double]) -> (Int, Double)? {
        guard let first = axis.first, let last = axis.last, axis.count >= 2,
              v >= first - 1e-12, v <= last + 1e-12 else { return nil }
        let h = (last - first) / Double(axis.count - 1)
        let pos = min(max((v - first) / h, 0), Double(axis.count - 1))
        let i = min(Int(pos), axis.count - 2)
        return (i, pos - Double(i))
    }

    static func step(_ axis: [Double]) -> Double {
        guard let first = axis.first, let last = axis.last, axis.count >= 2 else { return 1e-3 }
        return (last - first) / Double(axis.count - 1) / 2
    }
}
