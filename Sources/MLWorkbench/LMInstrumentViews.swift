import Charts
import SwiftUI
import WorkbenchCore

// MARK: Training dashboard

struct TrainingInstrumentView: View {
    let training: TrainingPayload
    @State private var smooth = true

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 18) {
                stat("step", training.points.last.map { "\($0.step)" } ?? "—")
                stat("loss", format(training.latestLoss))
                if let best = training.bestValLoss {
                    stat("best val", "\(format(best.loss)) @ \(best.step)")
                }
                if let tps = training.recentTokensPerSec {
                    stat("tokens/sec", tps >= 1000 ? String(format: "%.1fk", tps / 1000) : String(format: "%.0f", tps))
                }
                if !training.done {
                    ProgressView().controlSize(.small)
                }
                Spacer()
                Toggle("Smooth", isOn: $smooth)
                    .toggleStyle(.checkbox)
                    .font(.caption)
            }
            lossChart
                .frame(height: 200)
            HStack(alignment: .top, spacing: 12) {
                if !training.layers.isEmpty {
                    panel("gradient norm per layer") { gradChart }
                }
                if training.points.contains(where: { $0.lr != nil }) {
                    panel("learning rate") { lineChart(\.lr, name: "lr") }
                }
                if training.points.contains(where: { $0.tokensPerSec != nil }) {
                    panel("tokens/sec") { lineChart(\.tokensPerSec, name: "tokens/sec") }
                }
                ForEach(training.metricNames, id: \.self) { name in
                    panel(name) { metricChart(name) }
                }
            }
            if training.stride > 1 {
                Text("Showing every \(training.stride)th logged step.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.callout.monospacedDigit())
        }
    }

    private func panel<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content().frame(height: 120)
        }
        .frame(maxWidth: .infinity)
    }

    private var lossChart: some View {
        let train: [(step: Int, loss: Double)] = smooth
            ? training.smoothedLoss()
            : training.points.compactMap { p in p.loss.map { (step: p.step, loss: $0) } }
        let val: [(step: Int, loss: Double)] = training.points.compactMap { p in
            p.valLoss.map { (step: p.step, loss: $0) }
        }
        return Chart {
            ForEach(Array(train.enumerated()), id: \.offset) { _, p in
                LineMark(x: .value("step", p.step), y: .value("loss", p.loss))
                    .foregroundStyle(by: .value("split", "train"))
            }
            ForEach(Array(val.enumerated()), id: \.offset) { _, p in
                LineMark(x: .value("step", p.step), y: .value("loss", p.loss))
                    .foregroundStyle(by: .value("split", "validation"))
                PointMark(x: .value("step", p.step), y: .value("loss", p.loss))
                    .foregroundStyle(by: .value("split", "validation"))
                    .symbolSize(16)
            }
        }
        .chartXAxisLabel("step")
        .chartYAxisLabel("loss")
    }

    private var gradChart: some View {
        let rows: [(step: Int, layer: String, norm: Double)] = training.points.flatMap { p in
            (p.gradNorms ?? [:]).compactMap { name, norm in
                guard let norm, norm > 0 else { return nil }
                return (step: p.step, layer: name, norm: norm)
            }
        }
        return Chart(Array(rows.enumerated()), id: \.offset) { _, r in
            LineMark(x: .value("step", r.step), y: .value("norm", r.norm))
                .foregroundStyle(by: .value("layer", r.layer))
        }
        .chartYScale(type: .log)
        .chartLegend(position: .bottom)
    }

    private func lineChart(_ key: KeyPath<TrainingPayload.Point, Double?>, name: String) -> some View {
        let rows: [(step: Int, value: Double)] = training.points.compactMap { p in
            p[keyPath: key].map { (step: p.step, value: $0) }
        }
        return Chart(Array(rows.enumerated()), id: \.offset) { _, r in
            LineMark(x: .value("step", r.step), y: .value(name, r.value))
        }
    }

    private func metricChart(_ name: String) -> some View {
        let rows: [(step: Int, value: Double)] = training.points.compactMap { p in
            guard let v = p.metrics?[name] ?? nil else { return nil }
            return (step: p.step, value: v)
        }
        return Chart(Array(rows.enumerated()), id: \.offset) { _, r in
            LineMark(x: .value("step", r.step), y: .value(name, r.value))
        }
    }
}

// MARK: Embedding explorer

struct EmbeddingInstrumentView: View {
    let embedding: EmbeddingPayload
    @State private var hovered: EmbeddingPayload.Point?

    var body: some View {
        let points = embedding.plotted
        VStack(alignment: .leading, spacing: 6) {
            Chart(points) { p in
                PointMark(x: .value("component 1", p.x), y: .value("component 2", p.y))
                    .foregroundStyle(by: .value("class", p.tokenClass))
                    .symbolSize(hovered?.id == p.id ? 120 : 40)
                    .annotation(position: .top, spacing: 2) {
                        if points.count <= 80 || hovered?.id == p.id {
                            Text(p.label)
                                .font(.caption2.monospaced())
                                .foregroundStyle(hovered?.id == p.id ? .primary : .secondary)
                        }
                    }
            }
            .chartOverlay { proxy in
                GeometryReader { geo in
                    Rectangle().fill(.clear).contentShape(Rectangle())
                        .onContinuousHover { phase in
                            switch phase {
                            case .active(let location):
                                guard let frame = proxy.plotFrame else { return }
                                let origin = geo[frame].origin
                                let local = CGPoint(x: location.x - origin.x, y: location.y - origin.y)
                                if let value = proxy.value(at: local, as: (Double, Double).self) {
                                    hovered = embedding.nearest(x: value.0, y: value.1)
                                }
                            case .ended:
                                hovered = nil
                            }
                        }
                }
            }
            .frame(height: 320)
            Text(caption)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
        }
    }

    private var caption: String {
        var text = "\(embedding.labels.count) tokens × \(embedding.dim) dims, \(embedding.method.uppercased())"
        if let ev = embedding.explainedVariance, !ev.isEmpty {
            text += ", variance explained " + ev.map { String(format: "%.0f%%", $0 * 100) }.joined(separator: " + ")
        }
        if let hovered {
            text += "    \(hovered.label) (\(hovered.tokenClass)) at (\(format(hovered.x)), \(format(hovered.y)))"
        }
        return text
    }
}

// MARK: Heatmaps

/// A grid of colored cells with optional row and column labels, and a
/// hover readout. Used by the matrix instrument and the attention viewer.
struct LabeledHeatmap: View {
    let values: [[Double?]]
    let rowLabels: [String]?
    let colLabels: [String]?
    let range: ClosedRange<Double>
    var logScale = false
    var maxSide: CGFloat = 420
    var rowName = "row"
    var colName = "col"
    @State private var hover: (row: Int, col: Int)?

    private var rows: Int { values.count }
    private var cols: Int { values.first?.count ?? 0 }
    private var cell: CGFloat { max(4, min(22, maxSide / CGFloat(max(rows, cols, 1)))) }
    private var labelWidth: CGFloat { rowLabels == nil ? 0 : 44 }
    private var labelHeight: CGFloat { colLabels == nil ? 0 : 18 }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ScrollView([.horizontal, .vertical]) {
                Canvas { context, _ in draw(in: &context) }
                    .frame(width: labelWidth + CGFloat(cols) * cell, height: labelHeight + CGFloat(rows) * cell)
                    .contentShape(Rectangle())
                    .onContinuousHover { phase in
                        if case .active(let p) = phase {
                            let c = Int((p.x - labelWidth) / cell), r = Int((p.y - labelHeight) / cell)
                            if r >= 0, r < rows, c >= 0, c < cols, p.x >= labelWidth, p.y >= labelHeight {
                                hover = (row: r, col: c)
                            } else {
                                hover = nil
                            }
                        } else {
                            hover = nil
                        }
                    }
            }
            .frame(height: min(labelHeight + CGFloat(rows) * cell + 4, 460))
            Text(readout)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
        }
    }

    private var readout: String {
        guard let hover else { return "hover a cell" }
        let r = rowLabels.map { "\(rowName) \(hover.row) '\($0[hover.row])'" } ?? "\(rowName) \(hover.row)"
        let c = colLabels.map { "\(colName) \(hover.col) '\($0[hover.col])'" } ?? "\(colName) \(hover.col)"
        return "\(r), \(c): \(format(values[hover.row][hover.col]))"
    }

    private func draw(in context: inout GraphicsContext) {
        let fontSize = max(7, min(11, cell * 0.6))
        for r in 0..<rows {
            for c in 0..<min(cols, values[r].count) {
                let rect = CGRect(x: labelWidth + CGFloat(c) * cell, y: labelHeight + CGFloat(r) * cell,
                                  width: cell - 0.5, height: cell - 0.5)
                guard let v = values[r][c] else {
                    context.fill(Path(rect), with: .color(.gray.opacity(0.2)))
                    continue
                }
                let t = min(max(heatFraction(v, in: range, log: logScale), 0), 1)
                context.fill(Path(rect), with: .color(heatColor(t)))
            }
        }
        if let rowLabels, cell >= 7 {
            for (r, label) in rowLabels.prefix(rows).enumerated() {
                context.draw(Text(display(label)).font(.system(size: fontSize, design: .monospaced)),
                             at: CGPoint(x: labelWidth - 4, y: labelHeight + (CGFloat(r) + 0.5) * cell), anchor: .trailing)
            }
        }
        if let colLabels, cell >= 7 {
            for (c, label) in colLabels.prefix(cols).enumerated() {
                context.draw(Text(display(label)).font(.system(size: fontSize, design: .monospaced)),
                             at: CGPoint(x: labelWidth + (CGFloat(c) + 0.5) * cell, y: labelHeight - 3), anchor: .bottom)
            }
        }
        if let hover {
            let rect = CGRect(x: labelWidth + CGFloat(hover.col) * cell, y: labelHeight + CGFloat(hover.row) * cell,
                              width: cell, height: cell)
            context.stroke(Path(rect), with: .color(.primary), lineWidth: 1.5)
        }
    }

    /// Whitespace and newlines are invisible as labels.
    private func display(_ label: String) -> String {
        label.replacingOccurrences(of: "\n", with: "⏎").replacingOccurrences(of: " ", with: "␣")
    }
}

func heatColor(_ t: Double) -> Color {
    Color(hue: 0.62 - 0.5 * t, saturation: 0.15 + 0.7 * t, brightness: 0.97 - 0.45 * t)
}

struct MatrixInstrumentView: View {
    let matrix: MatrixPayload

    var body: some View {
        if let range = matrix.valueRange {
            LabeledHeatmap(values: matrix.values, rowLabels: matrix.rows, colLabels: matrix.cols,
                           range: range, logScale: matrix.logScale)
        } else {
            Text("No finite values to draw.").foregroundStyle(.secondary)
        }
    }
}

// MARK: Attention viewer

struct AttentionInstrumentView: View {
    @EnvironmentObject private var model: AppModel
    let attention: AttentionPayload
    @State private var layer = 0
    @State private var pinned: Int?
    @State private var hoveredHead: Int?
    @State private var prompt = ""

    private var heads: [[[Double?]]] {
        attention.layers.indices.contains(layer) ? attention.layers[layer].heads : []
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                if attention.layers.count > 1 {
                    Picker("Layer", selection: $layer) {
                        ForEach(attention.layers.indices, id: \.self) { i in
                            Text(attention.layers[i].name).tag(i)
                        }
                    }
                    .fixedSize()
                }
                if attention.callback != nil {
                    TextField("Type a prompt and press Return", text: $prompt)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit(run)
                    Button("Run", action: run)
                        .disabled(prompt.isEmpty || model.kernelState != .idle)
                }
            }
            ScrollView(.horizontal) {
                HStack(spacing: 8) {
                    ForEach(heads.indices, id: \.self) { h in
                        VStack(spacing: 2) {
                            Canvas { context, size in
                                drawThumbnail(heads[h], in: &context, size: size)
                            }
                            .frame(width: 72, height: 72)
                            .overlay(RoundedRectangle(cornerRadius: 2)
                                .stroke(pinned == h ? Color.accentColor : .clear, lineWidth: 2))
                            Text("head \(h)").font(.caption2)
                                .foregroundStyle(pinned == h ? .primary : .secondary)
                        }
                        .onTapGesture { pinned = pinned == h ? nil : h }
                        .onHover { hoveredHead = $0 ? h : (hoveredHead == h ? nil : hoveredHead) }
                        .help("Click to pin this head")
                    }
                }
                .padding(2)
            }
            let shown = hoveredHead ?? pinned ?? 0
            if heads.indices.contains(shown) {
                Text("\(attention.layers[layer].name), head \(shown)\(pinned == shown ? " (pinned)" : ""): row = query token, column = key token it attends to")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                LabeledHeatmap(values: heads[shown], rowLabels: attention.tokens, colLabels: attention.tokens,
                               range: 0...1, maxSide: 380, rowName: "query", colName: "key")
            }
        }
        .onAppear { prompt = attention.prompt ?? "" }
        .onChange(of: attention.prompt) { _, new in prompt = new ?? prompt }
        .onChange(of: attention.layers.count) { _, count in if layer >= count { layer = 0 } }
    }

    private func run() {
        guard !prompt.isEmpty else { return }
        model.rerun(attention: attention, prompt: prompt)
    }

    private func drawThumbnail(_ head: [[Double?]], in context: inout GraphicsContext, size: CGSize) {
        let n = max(head.count, 1)
        let w = size.width / CGFloat(n), h = size.height / CGFloat(n)
        for (r, row) in head.enumerated() {
            for (c, v) in row.enumerated() {
                guard let v else { continue }
                let rect = CGRect(x: CGFloat(c) * w, y: CGFloat(r) * h, width: w + 0.3, height: h + 0.3)
                context.fill(Path(rect), with: .color(heatColor(min(max(v, 0), 1))))
            }
        }
    }
}
