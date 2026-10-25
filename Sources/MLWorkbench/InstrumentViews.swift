import SwiftUI
import WorkbenchCore

/// The Instruments tab: every widget the learner's code published this
/// session, newest first.
struct InstrumentsPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        if model.instruments.isEmpty {
            ContentUnavailableView {
                Label("No instruments yet", systemImage: "waveform.path.ecg")
            } description: {
                Text("Call show_graph, show_landscape, show_training, show_embeddings, show_attention or show_matrix from your code, e.g. `from workbench import show_training`.")
            }
        } else {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    HStack {
                        Spacer()
                        Button("Clear") { model.clearInstruments() }
                            .buttonStyle(.borderless)
                    }
                    ForEach(model.instruments) { item in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(item.instrument.title)
                                    .font(.subheadline.weight(.semibold))
                                Spacer()
                                ExplainButton(summaries: item.explain)
                            }
                            switch item.instrument {
                            case .graph(let graph):
                                GraphInstrumentView(graph: graph)
                            case .landscape(let landscape):
                                LandscapeInstrumentView(landscape: landscape)
                            case .training(let training):
                                TrainingInstrumentView(training: training)
                            case .embedding(let embedding):
                                EmbeddingInstrumentView(embedding: embedding)
                            case .attention(let attention):
                                AttentionInstrumentView(attention: attention)
                            case .matrix(let matrix):
                                MatrixInstrumentView(matrix: matrix)
                            }
                        }
                        .padding(10)
                        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                    }
                }
                .padding(8)
            }
        }
    }
}

/// "Explain this tensor": asks the tutor about one of the widget's tensors.
struct ExplainButton: View {
    @EnvironmentObject private var model: AppModel
    let summaries: [JSONValue]

    var body: some View {
        if summaries.count == 1, let only = summaries.first {
            Button("Explain") { model.explain(only) }
                .buttonStyle(.borderless)
                .help("Ask the tutor about \(only.name ?? "this tensor")")
                .disabled(model.tutor.isAnswering)
        } else if !summaries.isEmpty {
            Menu("Explain") {
                ForEach(Array(summaries.enumerated()), id: \.offset) { _, summary in
                    Button(summary.name ?? "tensor") { model.explain(summary) }
                }
            }
            .menuStyle(.button)
            .buttonStyle(.borderless)
            .fixedSize()
            .disabled(model.tutor.isAnswering)
        }
    }
}

// MARK: Computation graph

struct GraphInstrumentView: View {
    let graph: GraphPayload
    @State private var step: Int?
    @State private var playing = false

    private let nodeSize = CGSize(width: 116, height: 50)
    private let gap = CGSize(width: 44, height: 16)

    private var currentStep: Int { step ?? graph.stepCount }

    var body: some View {
        let layout = graph.layout()
        let columns = (layout.values.map { $0.column }.max() ?? 0) + 1
        let rows = (layout.values.map { $0.row }.max() ?? 0) + 1
        let size = CGSize(
            width: CGFloat(columns) * (nodeSize.width + gap.width) + gap.width,
            height: CGFloat(rows) * (nodeSize.height + gap.height) + gap.height
        )
        VStack(alignment: .leading, spacing: 8) {
            controls
            ScrollView([.horizontal, .vertical]) {
                Canvas { context, _ in
                    draw(in: &context, layout: layout)
                }
                .frame(width: size.width, height: size.height)
            }
            .frame(height: min(size.height, 320))
        }
        .task(id: playing) {
            guard playing else { return }
            if currentStep >= graph.stepCount { step = 0 }
            while playing, currentStep < graph.stepCount {
                try? await Task.sleep(for: .milliseconds(550))
                if Task.isCancelled { return }
                step = currentStep + 1
            }
            playing = false
        }
    }

    private var controls: some View {
        HStack(spacing: 8) {
            Button {
                playing.toggle()
            } label: {
                Image(systemName: playing ? "pause.fill" : "play.fill")
            }
            .help(playing ? "Pause" : "Step through the forward and backward pass")
            Button {
                playing = false
                step = max(0, currentStep - 1)
            } label: {
                Image(systemName: "backward.frame")
            }
            Button {
                playing = false
                step = min(graph.stepCount, currentStep + 1)
            } label: {
                Image(systemName: "forward.frame")
            }
            Slider(
                value: Binding(
                    get: { Double(currentStep) },
                    set: { playing = false; step = Int($0.rounded()) }
                ),
                in: 0...Double(max(graph.stepCount, 1)),
                step: 1
            )
            .frame(maxWidth: 260)
            Text(phase)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.borderless)
    }

    private var phase: String {
        let n = graph.order.count
        if currentStep == 0 { return "Nothing computed yet" }
        if currentStep < n { return "Forward: computing values (\(currentStep)/\(n))" }
        if !graph.hasBackward { return "Values only. Call backward() first to see gradients." }
        if currentStep == n { return "Forward done. Backward starts at the output." }
        if currentStep < 2 * n { return "Backward: pushing gradients (\(currentStep - n)/\(n))" }
        return "All values and gradients"
    }

    private func frame(_ cell: (column: Int, row: Int)) -> CGRect {
        CGRect(
            x: gap.width + CGFloat(cell.column) * (nodeSize.width + gap.width),
            y: gap.height + CGFloat(cell.row) * (nodeSize.height + gap.height),
            width: nodeSize.width,
            height: nodeSize.height
        )
    }

    private func draw(in context: inout GraphicsContext, layout: [Int: (column: Int, row: Int)]) {
        let reveal = graph.reveal(at: currentStep)
        for edge in graph.edges where edge.count == 2 {
            guard let from = layout[edge[0]], let to = layout[edge[1]] else { continue }
            let a = frame(from), b = frame(to)
            let start = CGPoint(x: a.maxX, y: a.midY), end = CGPoint(x: b.minX, y: b.midY)
            var path = Path()
            path.move(to: start)
            path.addCurve(
                to: end,
                control1: CGPoint(x: start.x + gap.width / 2, y: start.y),
                control2: CGPoint(x: end.x - gap.width / 2, y: end.y)
            )
            let active = edge[1] == reveal.current || (reveal.grads.contains(edge[1]) && edge[0] == reveal.current)
            context.stroke(path, with: .color(active ? Color.accentColor : Color.secondary.opacity(0.5)),
                           lineWidth: active ? 2 : 1)
        }
        for node in graph.nodes {
            guard let cell = layout[node.id] else { continue }
            let rect = frame(cell)
            let shape = RoundedRectangle(cornerRadius: 7).path(in: rect)
            let isCurrent = node.id == reveal.current
            context.fill(shape, with: .color(node.op.isEmpty ? Color.blue.opacity(0.10) : Color.purple.opacity(0.10)))
            context.stroke(shape, with: .color(isCurrent ? Color.accentColor : Color.secondary.opacity(0.6)),
                           lineWidth: isCurrent ? 2.5 : 1)
            let title = [node.label, node.op.isEmpty ? "" : node.op]
                .filter { !$0.isEmpty }.joined(separator: "  ")
            context.draw(
                Text(title.isEmpty ? "input" : title).font(.caption.weight(.semibold)),
                at: CGPoint(x: rect.midX, y: rect.minY + 10)
            )
            let value = reveal.values.contains(node.id) ? format(node.data) : "?"
            context.draw(
                Text("data \(value)").font(.caption.monospaced()),
                at: CGPoint(x: rect.midX, y: rect.midY + 1)
            )
            if reveal.grads.contains(node.id) {
                context.draw(
                    Text("grad \(format(node.grad))").font(.caption.monospaced()).foregroundColor(.orange),
                    at: CGPoint(x: rect.midX, y: rect.maxY - 10)
                )
            }
        }
    }
}

func format(_ value: Double?) -> String {
    guard let value else { return "—" }
    if value != 0, abs(value) < 1e-3 || abs(value) >= 1e5 {
        return String(format: "%.3g", value)
    }
    return String(format: "%.4f", value)
        .replacingOccurrences(of: #"0+$"#, with: "", options: .regularExpression)
        .replacingOccurrences(of: #"\.$"#, with: "", options: .regularExpression)
}

// MARK: Loss landscape

struct LandscapeInstrumentView: View {
    let landscape: LandscapePayload
    /// Where the learner dragged the probe, in data coordinates.
    @State private var probe: (x: Double, y: Double)?

    private let inset = EdgeInsets(top: 10, leading: 52, bottom: 26, trailing: 12)

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            GeometryReader { geo in
                let plot = CGRect(
                    x: inset.leading, y: inset.top,
                    width: max(geo.size.width - inset.leading - inset.trailing, 10),
                    height: max(geo.size.height - inset.top - inset.bottom, 10)
                )
                Canvas { context, _ in
                    if landscape.dims == 2 {
                        drawHeatmap(in: &context, plot: plot)
                    } else {
                        drawLine(in: &context, plot: plot)
                    }
                    drawAxes(in: &context, plot: plot)
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in probe = toData(value.location, plot: plot) }
                )
            }
            .frame(height: landscape.dims == 2 ? 300 : 220)
            Text(readout)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    private var xRange: ClosedRange<Double> {
        (landscape.x.first ?? 0)...max(landscape.x.last ?? 1, (landscape.x.first ?? 0) + 1e-12)
    }

    private var yRange: ClosedRange<Double> {
        if landscape.dims == 2, let y = landscape.y {
            return (y.first ?? 0)...max(y.last ?? 1, (y.first ?? 0) + 1e-12)
        }
        let r = landscape.valueRange ?? 0...1
        let pad = (r.upperBound - r.lowerBound) * 0.05
        return (r.lowerBound - pad)...(r.upperBound + pad)
    }

    private func toScreen(_ x: Double, _ y: Double, plot: CGRect) -> CGPoint {
        let fx = (x - xRange.lowerBound) / (xRange.upperBound - xRange.lowerBound)
        let fy = (y - yRange.lowerBound) / (yRange.upperBound - yRange.lowerBound)
        return CGPoint(x: Double(plot.minX) + fx * Double(plot.width), y: Double(plot.maxY) - fy * Double(plot.height))
    }

    private func toData(_ p: CGPoint, plot: CGRect) -> (x: Double, y: Double) {
        let fx = min(max(Double((p.x - plot.minX) / plot.width), 0), 1)
        let fy = min(max(Double((plot.maxY - p.y) / plot.height), 0), 1)
        return (
            x: xRange.lowerBound + fx * (xRange.upperBound - xRange.lowerBound),
            y: yRange.lowerBound + fy * (yRange.upperBound - yRange.lowerBound)
        )
    }

    private var readout: String {
        let names = landscape.labels
        let pathNote: String
        if let last = landscape.path.last, let lastValue = last.last ?? nil {
            pathNote = "path: \(landscape.path.count) points, last f = \(format(lastValue))"
        } else {
            pathNote = "drag on the plot to probe"
        }
        guard let probe else { return pathNote }
        if landscape.dims == 2 {
            let f = landscape.value(atX: probe.x, y: probe.y)
            var text = "\(names.first ?? "x") = \(format(probe.x)), \(names.dropFirst().first ?? "y") = \(format(probe.y)), f = \(format(f))"
            if let g = landscape.gradient(atX: probe.x, y: probe.y) {
                text += ", gradient = (\(format(g[0])), \(format(g[1])))"
            }
            return text + "    " + pathNote
        }
        let f = landscape.value(atX: probe.x)
        var text = "\(names.first ?? "x") = \(format(probe.x)), f = \(format(f))"
        if let g = landscape.gradient(atX: probe.x) { text += ", slope = \(format(g[0]))" }
        return text + "    " + pathNote
    }

    // 1D

    private func drawLine(in context: inout GraphicsContext, plot: CGRect) {
        guard let line = landscape.line else { return }
        var path = Path()
        var penDown = false
        for (x, v) in zip(landscape.x, line) {
            guard let v else { penDown = false; continue }
            let p = toScreen(x, v, plot: plot)
            if penDown { path.addLine(to: p) } else { path.move(to: p) }
            penDown = true
        }
        context.stroke(path, with: .color(.accentColor), lineWidth: 2)
        drawPath(in: &context, plot: plot) { point in
            guard point.count >= 2, let x = point[0], let f = point[1] else { return nil }
            return toScreen(x, min(max(f, yRange.lowerBound), yRange.upperBound), plot: plot)
        }
        if let probe, let f = landscape.value(atX: probe.x) {
            let top = toScreen(probe.x, yRange.upperBound, plot: plot)
            let bottom = toScreen(probe.x, yRange.lowerBound, plot: plot)
            var guide = Path()
            guide.move(to: top)
            guide.addLine(to: bottom)
            context.stroke(guide, with: .color(.secondary), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            let p = toScreen(probe.x, f, plot: plot)
            if let slope = landscape.gradient(atX: probe.x)?.first {
                // Tangent line through the probe, a tenth of the x range wide.
                let half = (xRange.upperBound - xRange.lowerBound) / 10
                var tangent = Path()
                tangent.move(to: toScreen(probe.x - half, f - slope * half, plot: plot))
                tangent.addLine(to: toScreen(probe.x + half, f + slope * half, plot: plot))
                context.stroke(tangent, with: .color(.green), lineWidth: 1.5)
            }
            context.fill(Path(ellipseIn: CGRect(x: p.x - 4, y: p.y - 4, width: 8, height: 8)), with: .color(.green))
        }
    }

    // 2D

    private func drawHeatmap(in context: inout GraphicsContext, plot: CGRect) {
        guard let grid = landscape.grid, let ys = landscape.y, let range = landscape.valueRange else { return }
        let nx = landscape.x.count, ny = ys.count
        let cellW = plot.width / CGFloat(max(nx - 1, 1)), cellH = plot.height / CGFloat(max(ny - 1, 1))
        let span = range.upperBound - range.lowerBound
        for j in 0..<ny {
            for i in 0..<nx {
                guard let v = grid[j][i] else { continue }
                // Square-root scaling keeps detail near the minimum visible.
                let t = (span > 0 ? (v - range.lowerBound) / span : 0).squareRoot()
                let center = toScreen(landscape.x[i], ys[j], plot: plot)
                let rect = CGRect(x: center.x - cellW / 2, y: center.y - cellH / 2, width: cellW + 0.5, height: cellH + 0.5)
                    .intersection(plot)
                context.fill(Path(rect), with: .color(Color(hue: 0.66 - 0.66 * t, saturation: 0.75, brightness: 0.35 + 0.6 * t)))
            }
        }
        drawPath(in: &context, plot: plot) { point in
            guard point.count >= 2, let x = point[0], let y = point[1] else { return nil }
            return toScreen(x, y, plot: plot)
        }
        if let probe {
            let p = toScreen(probe.x, probe.y, plot: plot)
            context.stroke(Path(ellipseIn: CGRect(x: p.x - 5, y: p.y - 5, width: 10, height: 10)), with: .color(.white), lineWidth: 2)
            if let g = landscape.gradient(atX: probe.x, y: probe.y) {
                // Arrow downhill (against the gradient), drawn in screen space.
                let gx = g[0] / (xRange.upperBound - xRange.lowerBound) * Double(plot.width)
                let gy = -g[1] / (yRange.upperBound - yRange.lowerBound) * Double(plot.height)
                let length = max((gx * gx + gy * gy).squareRoot(), 1e-9)
                let tip = CGPoint(x: Double(p.x) - gx / length * 30, y: Double(p.y) - gy / length * 30)
                var arrow = Path()
                arrow.move(to: p)
                arrow.addLine(to: tip)
                context.stroke(arrow, with: .color(.white), lineWidth: 2)
                context.fill(Path(ellipseIn: CGRect(x: tip.x - 3, y: tip.y - 3, width: 6, height: 6)), with: .color(.white))
            }
        }
    }

    private func drawPath(in context: inout GraphicsContext, plot: CGRect, point: ([Double?]) -> CGPoint?) {
        let points = landscape.path.compactMap(point)
        guard !points.isEmpty else { return }
        var line = Path()
        line.addLines(points)
        context.stroke(line, with: .color(.orange), lineWidth: 1.5)
        for (k, p) in points.enumerated() {
            let r: CGFloat = k == points.count - 1 ? 5 : 3
            context.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)), with: .color(.orange))
        }
    }

    private func drawAxes(in context: inout GraphicsContext, plot: CGRect) {
        context.stroke(Path(plot), with: .color(.secondary.opacity(0.6)), lineWidth: 1)
        let label = { (s: String) in Text(s).font(.caption2.monospaced()).foregroundColor(.secondary) }
        context.draw(label(format(xRange.lowerBound)), at: CGPoint(x: plot.minX, y: plot.maxY + 12))
        context.draw(label(format(xRange.upperBound)), at: CGPoint(x: plot.maxX, y: plot.maxY + 12))
        context.draw(label(landscape.labels.first ?? "x"), at: CGPoint(x: plot.midX, y: plot.maxY + 12))
        context.draw(label(format(yRange.upperBound)), at: CGPoint(x: plot.minX - 4, y: plot.minY + 6), anchor: .trailing)
        context.draw(label(format(yRange.lowerBound)), at: CGPoint(x: plot.minX - 4, y: plot.maxY - 6), anchor: .trailing)
        let yName = landscape.dims == 2 ? (landscape.labels.dropFirst().first ?? "y") : "f"
        context.draw(label(yName), at: CGPoint(x: plot.minX - 4, y: plot.midY), anchor: .trailing)
    }
}
