import Foundation

/// A visual widget's data, as published by the `workbench` Python helper.
public enum Instrument: Equatable, Sendable {
    case graph(GraphPayload)
    case landscape(LandscapePayload)
    case training(TrainingPayload)
    case embedding(EmbeddingPayload)
    case attention(AttentionPayload)
    case matrix(MatrixPayload)
    case images(ImagesPayload)
    case features(FeaturesPayload)
    case attribution(AttributionPayload)
    case plot(PlotPayload)
    case timeline(TimelinePayload)

    public static let graphMIME = "application/vnd.workbench.graph+json"
    public static let landscapeMIME = "application/vnd.workbench.landscape+json"
    public static let trainingMIME = "application/vnd.workbench.training+json"
    public static let embeddingMIME = "application/vnd.workbench.embedding+json"
    public static let attentionMIME = "application/vnd.workbench.attention+json"
    public static let matrixMIME = "application/vnd.workbench.matrix+json"
    public static let imagesMIME = "application/vnd.workbench.images+json"
    public static let featuresMIME = "application/vnd.workbench.features+json"
    public static let attributionMIME = "application/vnd.workbench.attribution+json"
    public static let plotMIME = "application/vnd.workbench.plot+json"
    public static let timelineMIME = "application/vnd.workbench.timeline+json"

    /// Picks the instrument out of a display's MIME bundle, if it has one.
    public init?(data: [String: JSONValue]) {
        if let value = data[Self.graphMIME], let payload = try? value.decode(GraphPayload.self) {
            self = .graph(payload)
        } else if let value = data[Self.landscapeMIME], let payload = try? value.decode(LandscapePayload.self) {
            self = .landscape(payload)
        } else if let value = data[Self.trainingMIME], let payload = try? value.decode(TrainingPayload.self) {
            self = .training(payload)
        } else if let value = data[Self.embeddingMIME], let payload = try? value.decode(EmbeddingPayload.self) {
            self = .embedding(payload)
        } else if let value = data[Self.attentionMIME], let payload = try? value.decode(AttentionPayload.self) {
            self = .attention(payload)
        } else if let value = data[Self.matrixMIME], let payload = try? value.decode(MatrixPayload.self) {
            self = .matrix(payload)
        } else if let value = data[Self.imagesMIME], let payload = try? value.decode(ImagesPayload.self) {
            self = .images(payload)
        } else if let value = data[Self.featuresMIME], let payload = try? value.decode(FeaturesPayload.self) {
            self = .features(payload)
        } else if let value = data[Self.attributionMIME], let payload = try? value.decode(AttributionPayload.self) {
            self = .attribution(payload)
        } else if let value = data[Self.plotMIME], let payload = try? value.decode(PlotPayload.self) {
            self = .plot(payload)
        } else if let value = data[Self.timelineMIME], let payload = try? value.decode(TimelinePayload.self) {
            self = .timeline(payload)
        } else {
            return nil
        }
    }

    /// The tensor summaries a widget carries for "Explain this tensor"
    /// (the `explain` list every `workbench` payload has).
    public static func explainSummaries(data: [String: JSONValue]) -> [JSONValue] {
        for (key, value) in data.sorted(by: { $0.key < $1.key }) where key.hasPrefix("application/vnd.workbench.") {
            if case .object(let payload) = value, case .array(let summaries)? = payload["explain"] {
                return summaries
            }
        }
        return []
    }

    public var title: String {
        switch self {
        case .graph(let g): g.title ?? "Computation graph"
        case .landscape(let l): l.title ?? "Loss landscape"
        case .training(let t): t.title ?? "Training"
        case .embedding(let e): e.title ?? "Embeddings"
        case .attention(let a): a.title ?? "Attention"
        case .matrix(let m): m.title ?? "Matrix"
        case .images(let i): i.title ?? "Images"
        case .features(let f): f.title ?? "Features"
        case .attribution(let a): a.title ?? "Logit attribution"
        case .plot(let p): p.title ?? "Plot"
        case .timeline(let t): t.title ?? "Timeline"
        }
    }
}

extension JSONValue {
    /// The `name` field of an object, if it has one.
    public var name: String? {
        if case .object(let o) = self, case .string(let s)? = o["name"] { return s }
        return nil
    }

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

// MARK: Training dashboard

public struct TrainingPayload: Codable, Equatable, Sendable {
    public struct Point: Codable, Equatable, Sendable {
        public var step: Int
        public var loss: Double?
        public var valLoss: Double?
        public var lr: Double?
        public var gradNorms: [String: Double?]?
        public var tokensPerSec: Double?
        public var metrics: [String: Double?]?

        enum CodingKeys: String, CodingKey {
            case step, loss, lr, metrics
            case valLoss = "val_loss"
            case gradNorms = "grad_norms"
            case tokensPerSec = "tokens_per_sec"
        }
    }

    public var title: String?
    public var points: [Point]
    /// Every `stride`-th logged step is kept once the curve gets long.
    public var stride: Int
    public var done: Bool

    /// Names of the layers that reported gradient norms, in first-seen order.
    public var layers: [String] {
        var seen: [String] = []
        for point in points {
            for name in (point.gradNorms ?? [:]).keys.sorted() where !seen.contains(name) {
                seen.append(name)
            }
        }
        return seen
    }

    public var metricNames: [String] {
        Array(Set(points.flatMap { ($0.metrics ?? [:]).keys })).sorted()
    }

    public var latestLoss: Double? { points.last(where: { $0.loss != nil })?.loss }
    public var bestValLoss: (step: Int, loss: Double)? {
        let values: [(step: Int, loss: Double)] = points.compactMap { p in
            guard let v = p.valLoss else { return nil }
            return (step: p.step, loss: v)
        }
        return values.min { $0.loss < $1.loss }
    }

    /// Mean of the most recent tokens/sec readings: one is noisy.
    public var recentTokensPerSec: Double? {
        let values = points.compactMap(\.tokensPerSec).suffix(20)
        return values.isEmpty ? nil : values.reduce(0, +) / Double(values.count)
    }

    /// Loss smoothed with an exponential moving average, for a readable curve.
    public func smoothedLoss(weight: Double = 0.9) -> [(step: Int, loss: Double)] {
        var out: [(step: Int, loss: Double)] = []
        var average: Double?
        for point in points {
            guard let loss = point.loss else { continue }
            let next = average.map { weight * $0 + (1 - weight) * loss } ?? loss
            average = next
            out.append((step: point.step, loss: next))
        }
        return out
    }
}

// MARK: Embedding explorer

public struct EmbeddingPayload: Codable, Equatable, Sendable {
    public var title: String?
    public var method: String
    /// One [x, y] per token; null where the projection wasn't finite.
    public var points: [[Double?]]
    public var labels: [String]
    public var classes: [String]
    public var explainedVariance: [Double]?
    public var dim: Int

    enum CodingKeys: String, CodingKey {
        case title, method, points, labels, classes, dim
        case explainedVariance = "explained_variance"
    }

    public struct Point: Identifiable, Equatable, Sendable {
        public var id: Int
        public var x: Double
        public var y: Double
        public var label: String
        public var tokenClass: String
    }

    public var plotted: [Point] {
        points.enumerated().compactMap { i, p in
            guard p.count >= 2, let x = p[0], let y = p[1] else { return nil }
            return Point(id: i, x: x, y: y, label: i < labels.count ? labels[i] : "\(i)",
                         tokenClass: i < classes.count ? classes[i] : "token")
        }
    }

    /// The point nearest (x, y), for hover.
    public func nearest(x: Double, y: Double) -> Point? {
        plotted.min { ($0.x - x) * ($0.x - x) + ($0.y - y) * ($0.y - y)
            < ($1.x - x) * ($1.x - x) + ($1.y - y) * ($1.y - y) }
    }
}

// MARK: Attention viewer

public struct AttentionPayload: Codable, Equatable, Sendable {
    public struct Layer: Codable, Equatable, Sendable {
        public var name: String
        /// heads[h][query][key]
        public var heads: [[[Double?]]]
    }

    public var title: String?
    public var prompt: String?
    public var tokens: [String]
    public var layers: [Layer]
    /// Set when the viewer can re-run the model on a new prompt.
    public var callback: String?

    /// The kernel code that re-runs this viewer on `prompt`.
    public func rerunCode(prompt: String) -> String? {
        guard let callback else { return nil }
        let encoder = JSONEncoder()
        // "\/" isn't a Python escape, so keep slashes as they are.
        encoder.outputFormatting = .withoutEscapingSlashes
        guard let id = try? encoder.encode(callback), let text = try? encoder.encode(prompt) else { return nil }
        // A JSON string is a valid Python string literal.
        return "import workbench as _wb; _wb.rerun(\(String(decoding: id, as: UTF8.self)), \(String(decoding: text, as: UTF8.self)))"
    }
}

// MARK: Heatmap

public struct MatrixPayload: Codable, Equatable, Sendable {
    public var title: String?
    public var values: [[Double?]]
    public var rows: [String]?
    public var cols: [String]?
    public var logScale: Bool

    enum CodingKeys: String, CodingKey {
        case title, values, rows, cols
        case logScale = "log_scale"
    }

    public var valueRange: ClosedRange<Double>? {
        let all = values.flatMap { $0.compactMap { $0 } }
        guard let lo = all.min(), let hi = all.max() else { return nil }
        return lo...max(hi, lo + 1e-12)
    }
}

/// Maps a value to 0...1 for coloring, optionally on a log scale
/// (log(1 + v - min)), which keeps small counts visible next to large ones.
public func heatFraction(_ value: Double, in range: ClosedRange<Double>, log: Bool) -> Double {
    let span = range.upperBound - range.lowerBound
    guard span > 0 else { return 0 }
    if log {
        return Foundation.log1p(value - range.lowerBound) / Foundation.log1p(span)
    }
    return (value - range.lowerBound) / span
}

// MARK: Image grid

public struct ImagesPayload: Codable, Equatable, Sendable {
    public struct Image: Codable, Equatable, Sendable {
        public var label: String?
        public var width: Int
        public var height: Int
        /// 1 (gray) or 3 (RGB, interleaved).
        public var channels: Int
        /// Base64 of width × height × channels bytes, row-major.
        public var pixels: String
        /// The raw values' min and max before scaling to 0...255.
        public var range: [Double]?

        /// Pixel bytes, or nil if they don't match the declared size.
        public var bytes: Data? {
            guard let data = Data(base64Encoded: pixels), data.count == width * height * channels else {
                return nil
            }
            return data
        }
    }

    public var title: String?
    public var images: [Image]
}

// MARK: Feature browser

public struct FeaturesPayload: Codable, Equatable, Sendable {
    public struct TokenValue: Codable, Equatable, Sendable {
        public var token: String
        public var value: Double
    }

    public struct Example: Codable, Equatable, Sendable {
        public var tokens: [String]
        public var acts: [Double?]
        /// The token the example was picked for (the activation peak).
        public var highlight: Int
    }

    public struct Feature: Codable, Equatable, Sendable, Identifiable {
        public var id: Int
        public var label: String?
        public var density: Double?
        public var maxAct: Double
        public var examples: [Example]
        public var up: [TokenValue]
        public var down: [TokenValue]

        enum CodingKeys: String, CodingKey {
            case id, label, density, examples, up, down
            case maxAct = "max_act"
        }

        public var name: String { label.map { "#\(id) \($0)" } ?? "#\(id)" }
    }

    public var title: String?
    public var features: [Feature]
}

// MARK: Logit attribution

public struct AttributionPayload: Codable, Equatable, Sendable {
    public struct Target: Codable, Equatable, Sendable {
        public var pos: Int
        public var token: String
    }

    public struct Node: Codable, Equatable, Sendable {
        public var component: Int
        public var pos: Int
        public var effect: Double
    }

    public struct Edge: Codable, Equatable, Sendable {
        public var source: Int
        public var target: Int
        public var weight: Double
    }

    public struct Graph: Codable, Equatable, Sendable {
        public var nodes: [Node]
        public var edges: [Edge]
    }

    public var title: String?
    public var tokens: [String]
    public var components: [String]
    /// values[component][token]
    public var values: [[Double?]]
    public var target: Target?
    public var graph: Graph?

    /// The largest |value|, for a symmetric color scale (at least a tiny
    /// positive number so all-zero input still draws).
    public var maxMagnitude: Double {
        max(values.flatMap { $0.compactMap { $0.map(abs) } }.max() ?? 0, 1e-12)
    }

    /// Each token's total over components.
    public var totals: [Double] {
        tokens.indices.map { t in values.reduce(0) { $0 + (t < $1.count ? ($1[t] ?? 0) : 0) } }
    }
}

/// -1...1 for a diverging color scale centred on 0.
public func signedFraction(_ value: Double, magnitude: Double) -> Double {
    guard magnitude > 0 else { return 0 }
    return min(max(value / magnitude, -1), 1)
}

// MARK: Plot

public struct PlotSeries: Codable, Equatable, Sendable {
    public var name: String
    /// "line" or "scatter".
    public var kind: String
    public var x: [Double?]
    public var y: [Double?]

    public struct Point: Equatable, Sendable {
        public var x: Double
        public var y: Double
    }

    /// The finite (x, y) pairs, skipping any that a log axis can't show.
    public func points(logX: Bool = false, logY: Bool = false) -> [Point] {
        zip(x, y).compactMap { px, py in
            guard let px, let py, px.isFinite, py.isFinite else { return nil }
            if (logX && px <= 0) || (logY && py <= 0) { return nil }
            return Point(x: px, y: py)
        }
    }
}

public struct PlotPayload: Codable, Equatable, Sendable {
    public var title: String?
    public var xLabel: String?
    public var yLabel: String?
    public var logX: Bool
    public var logY: Bool
    public var series: [PlotSeries]

    enum CodingKeys: String, CodingKey {
        case title, series
        case xLabel = "x_label"
        case yLabel = "y_label"
        case logX = "log_x"
        case logY = "log_y"
    }
}

// MARK: Timeline

public struct TimelinePayload: Codable, Equatable, Sendable {
    public struct Span: Codable, Equatable, Sendable {
        public var lane: Int
        public var start: Double
        public var end: Double
        public var label: String?
        public var kind: String?
    }

    public var title: String?
    public var unit: String
    public var lanes: [String]
    public var spans: [Span]
    public var metrics: [String: Double]
    public var below: [PlotSeries]

    /// The time range covered by the spans and the series below.
    public var timeRange: ClosedRange<Double> {
        let starts = spans.map(\.start) + below.flatMap { $0.points().map(\.x) }
        let ends = spans.map(\.end) + below.flatMap { $0.points().map(\.x) }
        let lo = starts.min() ?? 0, hi = ends.max() ?? 1
        return lo...max(hi, lo + 1e-9)
    }

    /// Distinct span kinds in first-seen order, for a stable color legend.
    public var kinds: [String] {
        var seen: [String] = []
        for k in spans.compactMap(\.kind) where !seen.contains(k) { seen.append(k) }
        return seen
    }

    /// Fraction of lane-time covered by spans: utilization across lanes.
    public var utilization: Double {
        let range = timeRange
        let total = (range.upperBound - range.lowerBound) * Double(max(lanes.count, 1))
        guard total > 0 else { return 0 }
        return spans.reduce(0) { $0 + ($1.end - $1.start) } / total
    }
}
