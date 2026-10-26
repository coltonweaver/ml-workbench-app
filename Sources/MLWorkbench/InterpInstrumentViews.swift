import SwiftUI
import WorkbenchCore

/// Red for positive, blue for negative, white at zero. `t` is -1...1.
func divergingColor(_ t: Double) -> Color {
    let m = min(abs(t), 1)
    return t >= 0
        ? Color(red: 1, green: 1 - 0.75 * m, blue: 1 - 0.8 * m)
        : Color(red: 1 - 0.8 * m, green: 1 - 0.6 * m, blue: 1)
}

/// Tokens as a flowing row of chips, each shaded by a value.
struct HighlightedTokens: View {
    let tokens: [String]
    let values: [Double?]
    let magnitude: Double
    var highlight: Int?
    @Binding var hover: Int?

    var body: some View {
        FlowLayout(spacing: 1) {
            ForEach(tokens.indices, id: \.self) { i in
                let v = i < values.count ? values[i] : nil
                Text(display(tokens[i]))
                    .font(.system(.callout, design: .monospaced))
                    .padding(.horizontal, 2)
                    .padding(.vertical, 1)
                    .background(divergingColor(signedFraction(v ?? 0, magnitude: magnitude)))
                    .overlay(alignment: .bottom) {
                        if i == highlight {
                            Rectangle().fill(Theme.accent).frame(height: 2)
                        }
                    }
                    .foregroundStyle(.black)
                    .onHover { inside in hover = inside ? i : (hover == i ? nil : hover) }
            }
        }
    }
}

/// Shows whitespace so a token's boundaries are visible.
func display(_ token: String) -> String {
    token.replacingOccurrences(of: " ", with: "·").replacingOccurrences(of: "\n", with: "⏎")
}

/// A minimal wrapping layout for token chips.
struct FlowLayout: Layout {
    var spacing: CGFloat = 2

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(width: proposal.width ?? 600, subviews: subviews)
        return CGSize(width: rows.width, height: rows.height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = arrange(width: bounds.width, subviews: subviews)
        for (i, origin) in rows.origins.enumerated() {
            subviews[i].place(at: CGPoint(x: bounds.minX + origin.x, y: bounds.minY + origin.y),
                              proposal: .unspecified)
        }
    }

    private func arrange(width: CGFloat, subviews: Subviews) -> (origins: [CGPoint], width: CGFloat, height: CGFloat) {
        var origins: [CGPoint] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        var widest: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x > 0, x + size.width > width {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            origins.append(CGPoint(x: x, y: y))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
            widest = max(widest, x)
        }
        return (origins, widest, y + rowHeight)
    }
}

/// Pick a feature on the left; on the right, the contexts where it fires
/// most (each token shaded by the feature's activation) and its logit effects.
struct FeatureBrowserView: View {
    let payload: FeaturesPayload
    @State private var selected: Int?
    @State private var filter = ""
    @State private var hover: (example: Int, token: Int)?

    private var shown: [FeaturesPayload.Feature] {
        guard !filter.isEmpty else { return payload.features }
        return payload.features.filter { f in
            f.name.localizedCaseInsensitiveContains(filter)
                || f.examples.contains { $0.tokens.joined().localizedCaseInsensitiveContains(filter) }
        }
    }

    private var current: FeaturesPayload.Feature? {
        payload.features.first { $0.id == selected } ?? payload.features.first
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                TextField("Filter by id, label or text", text: $filter)
                    .textFieldStyle(.roundedBorder)
                    .controlSize(.small)
                List(shown, selection: $selected) { f in
                    HStack {
                        Text(f.name).font(.caption.monospaced())
                        Spacer()
                        if let d = f.density {
                            Text(String(format: "%.2g%%", d * 100)).font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    .tag(f.id)
                }
                .listStyle(.bordered)
            }
            .frame(width: 180, height: 420)

            if let f = current {
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Feature \(f.name)").font(.headline)
                            Spacer()
                            Text(summary(f)).font(.caption).foregroundStyle(.secondary)
                        }
                        if !f.up.isEmpty || !f.down.isEmpty {
                            HStack(alignment: .top, spacing: 24) {
                                tokenList("Promotes", f.up)
                                tokenList("Suppresses", f.down)
                            }
                        }
                        Text("Top activations").font(.subheadline)
                        ForEach(f.examples.indices, id: \.self) { e in
                            let example = f.examples[e]
                            HighlightedTokens(tokens: example.tokens, values: example.acts,
                                              magnitude: max(f.maxAct, 1e-12), highlight: example.highlight,
                                              hover: binding(for: e))
                            Divider()
                        }
                        Text(readout(f)).font(.caption.monospaced()).foregroundStyle(.secondary)
                    }
                    .padding(.trailing, 8)
                }
                .frame(height: 420)
            } else {
                Text("No features.").foregroundStyle(.secondary)
            }
        }
    }

    private func summary(_ f: FeaturesPayload.Feature) -> String {
        var text = "max " + format(f.maxAct)
        if let d = f.density {
            text += ", fires on " + String(format: "%.2g%%", d * 100) + " of tokens"
        }
        return text
    }

    private func binding(for example: Int) -> Binding<Int?> {
        Binding(
            get: { hover?.example == example ? hover?.token : nil },
            set: { token in
                if let token {
                    hover = (example: example, token: token)
                } else if hover?.example == example {
                    hover = nil
                }
            })
    }

    private func readout(_ f: FeaturesPayload.Feature) -> String {
        guard let hover, f.examples.indices.contains(hover.example) else { return "hover a token" }
        let ex = f.examples[hover.example]
        guard ex.tokens.indices.contains(hover.token) else { return "hover a token" }
        let act = hover.token < ex.acts.count ? ex.acts[hover.token] : nil
        return "'\(display(ex.tokens[hover.token]))': \(format(act))"
    }

    private func tokenList(_ title: String, _ items: [FeaturesPayload.TokenValue]) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            ForEach(items.indices, id: \.self) { i in
                HStack(spacing: 6) {
                    Text(display(items[i].token)).font(.caption.monospaced())
                    Text(format(items[i].value)).font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }
}

/// Logit attribution: a component × token heatmap (red adds to the target
/// logit, blue takes away), and the attribution graph when there is one.
struct AttributionInstrumentView: View {
    let payload: AttributionPayload
    @State private var hover: (row: Int, col: Int)?
    @State private var showGraph = true

    private let cell: CGFloat = 26
    private let labelWidth: CGFloat = 56

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let target = payload.target {
                Text("Target: '\(display(target.token))' at position \(target.pos)")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if payload.graph != nil {
                Picker("", selection: $showGraph) {
                    Text("Graph").tag(true)
                    Text("Heatmap").tag(false)
                }
                .pickerStyle(.segmented)
                .frame(width: 200)
            }
            if let graph = payload.graph, showGraph {
                AttributionGraphView(payload: payload, graph: graph)
            } else {
                heatmap
            }
        }
    }

    private var heatmap: some View {
        let mag = payload.maxMagnitude
        let rows = payload.components.count, cols = payload.tokens.count
        return VStack(alignment: .leading, spacing: 4) {
            ScrollView(.horizontal) {
                Canvas { context, _ in
                    for (c, token) in payload.tokens.enumerated() {
                        let x = labelWidth + CGFloat(c) * cell + cell / 2
                        context.draw(Text(display(token)).font(.caption.monospaced()), at: CGPoint(x: x, y: 9))
                    }
                    for r in 0..<rows {
                        let y = 20 + CGFloat(r) * cell
                        context.draw(Text(payload.components[r]).font(.caption.monospaced()),
                                     at: CGPoint(x: labelWidth - 4, y: y + cell / 2), anchor: .trailing)
                        for c in 0..<min(cols, payload.values[r].count) {
                            let rect = CGRect(x: labelWidth + CGFloat(c) * cell, y: y, width: cell - 1, height: cell - 1)
                            var color = Color.gray.opacity(0.2)
                            if let v = payload.values[r][c] {
                                color = divergingColor(signedFraction(v, magnitude: mag))
                            }
                            context.fill(Path(rect), with: .color(color))
                        }
                    }
                    if let target = payload.target, target.pos < cols {
                        let rect = CGRect(x: labelWidth + CGFloat(target.pos) * cell, y: 20,
                                          width: cell - 1, height: CGFloat(rows) * cell - 1)
                        context.stroke(Path(rect), with: .color(Theme.accent), lineWidth: 2)
                    }
                }
                .frame(width: labelWidth + CGFloat(cols) * cell, height: 22 + CGFloat(rows) * cell)
                .onContinuousHover { phase in
                    if case .active(let p) = phase {
                        let c = Int((p.x - labelWidth) / cell), r = Int((p.y - 20) / cell)
                        if r >= 0, r < rows, c >= 0, c < cols, p.x >= labelWidth, p.y >= 20 {
                            hover = (row: r, col: c)
                        } else {
                            hover = nil
                        }
                    } else {
                        hover = nil
                    }
                }
            }
            Text(readout).font(.caption.monospaced()).foregroundStyle(.secondary)
        }
    }

    private var readout: String {
        guard let hover else {
            return "hover a cell · total at each token: " + payload.totals.map { format($0) }.joined(separator: " ")
        }
        let v = hover.col < payload.values[hover.row].count ? payload.values[hover.row][hover.col] : nil
        return "\(payload.components[hover.row]) at '\(display(payload.tokens[hover.col]))' (\(hover.col)): \(format(v))"
    }
}

/// Nodes laid out with position on x and component (layer order) on y,
/// bottom to top; edge width and color show the weight.
struct AttributionGraphView: View {
    let payload: AttributionPayload
    let graph: AttributionPayload.Graph
    @State private var selected: Int?

    private let colWidth: CGFloat = 56
    private let rowHeight: CGFloat = 54

    private var components: [Int] { Array(Set(graph.nodes.map(\.component))).sorted() }
    private var positions: [Int] { Array(Set(graph.nodes.map(\.pos))).sorted() }

    private func point(_ node: AttributionPayload.Node) -> CGPoint {
        let col = positions.firstIndex(of: node.pos) ?? 0
        let row = components.firstIndex(of: node.component) ?? 0
        return CGPoint(x: 40 + CGFloat(col) * colWidth,
                       y: 30 + CGFloat(components.count - 1 - row) * rowHeight)
    }

    var body: some View {
        let maxEffect = max(graph.nodes.map { abs($0.effect) }.max() ?? 0, 1e-12)
        let maxWeight = max(graph.edges.map { abs($0.weight) }.max() ?? 0, 1e-12)
        let width = 80 + CGFloat(max(positions.count - 1, 0)) * colWidth
        let height = 70 + CGFloat(max(components.count - 1, 0)) * rowHeight
        VStack(alignment: .leading, spacing: 6) {
            ScrollView([.horizontal, .vertical]) {
                ZStack(alignment: .topLeading) {
                    Canvas { context, _ in
                        for edge in graph.edges where graph.nodes.indices.contains(edge.source)
                            && graph.nodes.indices.contains(edge.target) {
                            let a = point(graph.nodes[edge.source]), b = point(graph.nodes[edge.target])
                            var path = Path()
                            path.move(to: a)
                            path.addCurve(to: b, control1: CGPoint(x: a.x, y: (a.y + b.y) / 2),
                                          control2: CGPoint(x: b.x, y: (a.y + b.y) / 2))
                            let t = signedFraction(edge.weight, magnitude: maxWeight)
                            let touches = selected == nil || selected == edge.source || selected == edge.target
                            let base: Color = t >= 0 ? .red : .blue
                            let opacity = touches ? 0.25 + 0.6 * abs(t) : 0.06
                            context.stroke(path, with: .color(base.opacity(opacity)), lineWidth: 0.5 + 4 * abs(t))
                        }
                        for (i, node) in graph.nodes.enumerated() {
                            let p = point(node)
                            let r: CGFloat = 7 + 9 * CGFloat(abs(node.effect) / maxEffect)
                            let rect = CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)
                            context.fill(Path(ellipseIn: rect), with: .color(divergingColor(signedFraction(node.effect, magnitude: maxEffect))))
                            context.stroke(Path(ellipseIn: rect), with: .color(selected == i ? Theme.accent : .secondary),
                                           lineWidth: selected == i ? 2 : 0.5)
                            let label = node.component < payload.components.count ? payload.components[node.component] : "?"
                            context.draw(Text(label).font(.caption2.monospaced()).foregroundColor(.black), at: p)
                        }
                        for (col, pos) in positions.enumerated() where pos < payload.tokens.count {
                            context.draw(Text(display(payload.tokens[pos])).font(.caption.monospaced()),
                                         at: CGPoint(x: 40 + CGFloat(col) * colWidth, y: height - 10))
                        }
                    }
                    .frame(width: width, height: height)
                    .contentShape(Rectangle())
                    .onTapGesture { location in selected = nearestNode(to: location) }
                }
            }
            .frame(height: min(height + 8, 460))
            Text(readout).font(.caption.monospaced()).foregroundStyle(.secondary)
        }
    }

    private func nearestNode(to location: CGPoint) -> Int? {
        var best: Int?
        var bestDistance: CGFloat = 20
        for i in graph.nodes.indices {
            let p = point(graph.nodes[i])
            let d = hypot(p.x - location.x, p.y - location.y)
            if d < bestDistance {
                best = i
                bestDistance = d
            }
        }
        return best
    }

    private var readout: String {
        guard let selected, graph.nodes.indices.contains(selected) else {
            return "\(graph.nodes.count) nodes, \(graph.edges.count) edges · click a node"
        }
        let node = graph.nodes[selected]
        let name = node.component < payload.components.count ? payload.components[node.component] : "?"
        let token = node.pos < payload.tokens.count ? display(payload.tokens[node.pos]) : "?"
        let incoming = graph.edges.filter { $0.target == selected }.count
        let outgoing = graph.edges.filter { $0.source == selected }.count
        return "\(name) at '\(token)' (\(node.pos)): effect \(format(node.effect)), \(incoming) in, \(outgoing) out"
    }
}
