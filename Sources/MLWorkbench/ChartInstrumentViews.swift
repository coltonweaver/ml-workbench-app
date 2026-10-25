import Charts
import SwiftUI
import WorkbenchCore

private let palette: [Color] = [.blue, .orange, .green, .pink, .purple, .teal, .brown, .red, .indigo, .mint]

/// Lines and points on shared axes, optionally log-scaled.
struct PlotInstrumentView: View {
    let payload: PlotPayload

    var body: some View {
        SeriesChart(series: payload.series, logX: payload.logX, logY: payload.logY,
                    xLabel: payload.xLabel, yLabel: payload.yLabel)
            .frame(height: 260)
    }
}

struct SeriesChart: View {
    let series: [PlotSeries]
    var logX = false
    var logY = false
    var xLabel: String?
    var yLabel: String?
    var xDomain: ClosedRange<Double>?

    var body: some View {
        let chart = Chart {
            ForEach(series.indices, id: \.self) { i in
                let s = series[i]
                ForEach(Array(s.points(logX: logX, logY: logY).enumerated()), id: \.offset) { _, p in
                    if s.kind == "scatter" {
                        PointMark(x: .value(xLabel ?? "x", p.x), y: .value(yLabel ?? "y", p.y))
                            .foregroundStyle(by: .value("series", s.name))
                    } else {
                        LineMark(x: .value(xLabel ?? "x", p.x), y: .value(yLabel ?? "y", p.y))
                            .foregroundStyle(by: .value("series", s.name))
                    }
                }
            }
        }
        .chartXAxisLabel(xLabel ?? "")
        .chartYAxisLabel(yLabel ?? "")
        .chartLegend(series.count > 1 ? .visible : .hidden)

        switch (logX, logY) {
        case (true, true): chart.chartXScale(type: .log).chartYScale(type: .log)
        case (true, false): chart.chartXScale(type: .log)
        case (false, true): chart.chartYScale(type: .log)
        case (false, false):
            if let xDomain {
                chart.chartXScale(domain: xDomain)
            } else {
                chart
            }
        }
    }
}

/// Lanes of spans on a shared time axis: pipeline schedules, request
/// lifetimes in the serving simulator. Hover a span for its details.
struct TimelineInstrumentView: View {
    let payload: TimelinePayload
    @State private var hovered: Int?

    private let laneHeight: CGFloat = 20
    private let labelWidth: CGFloat = 90

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !payload.metrics.isEmpty {
                HStack(spacing: 18) {
                    ForEach(payload.metrics.keys.sorted(), id: \.self) { key in
                        VStack(alignment: .leading, spacing: 1) {
                            Text(key).font(.caption2).foregroundStyle(.secondary)
                            Text(format(payload.metrics[key])).font(.callout.monospacedDigit())
                        }
                    }
                }
            }
            if !payload.kinds.isEmpty {
                HStack(spacing: 10) {
                    ForEach(Array(payload.kinds.enumerated()), id: \.offset) { i, kind in
                        HStack(spacing: 4) {
                            RoundedRectangle(cornerRadius: 2).fill(palette[i % palette.count]).frame(width: 10, height: 10)
                            Text(kind).font(.caption)
                        }
                    }
                }
            }
            GeometryReader { geo in
                ScrollView(.vertical) {
                    Canvas { context, size in draw(in: &context, width: size.width) }
                        .frame(width: geo.size.width, height: CGFloat(payload.lanes.count) * laneHeight + 18)
                        .onContinuousHover { phase in
                            if case .active(let p) = phase {
                                hovered = span(at: p, width: geo.size.width)
                            } else {
                                hovered = nil
                            }
                        }
                }
            }
            .frame(height: min(CGFloat(payload.lanes.count) * laneHeight + 22, 360))
            Text(readout).font(.caption.monospaced()).foregroundStyle(.secondary)
            if !payload.below.isEmpty {
                SeriesChart(series: payload.below, xLabel: payload.unit, xDomain: payload.timeRange)
                    .frame(height: 140)
                    .padding(.leading, labelWidth - 30)
            }
        }
    }

    private func x(_ t: Double, width: CGFloat) -> CGFloat {
        let r = payload.timeRange
        return labelWidth + CGFloat((t - r.lowerBound) / (r.upperBound - r.lowerBound)) * (width - labelWidth - 8)
    }

    private func rect(_ s: TimelinePayload.Span, width: CGFloat) -> CGRect {
        let x0 = x(s.start, width: width), x1 = x(s.end, width: width)
        return CGRect(x: x0, y: CGFloat(s.lane) * laneHeight + 2, width: max(x1 - x0, 1), height: laneHeight - 4)
    }

    private func color(_ s: TimelinePayload.Span) -> Color {
        guard let kind = s.kind, let i = payload.kinds.firstIndex(of: kind) else { return .gray }
        return palette[i % palette.count]
    }

    private func draw(in context: inout GraphicsContext, width: CGFloat) {
        for (i, lane) in payload.lanes.enumerated() {
            let y = CGFloat(i) * laneHeight + laneHeight / 2
            context.draw(Text(lane).font(.caption.monospaced()), at: CGPoint(x: labelWidth - 6, y: y), anchor: .trailing)
            if i % 2 == 1 {
                context.fill(Path(CGRect(x: labelWidth, y: CGFloat(i) * laneHeight, width: width - labelWidth,
                                         height: laneHeight)), with: .color(.gray.opacity(0.06)))
            }
        }
        for (i, s) in payload.spans.enumerated() {
            let r = rect(s, width: width)
            context.fill(Path(roundedRect: r, cornerRadius: 3), with: .color(color(s).opacity(hovered == i ? 1 : 0.75)))
            if let label = s.label, r.width > 18 {
                context.draw(Text(label).font(.caption2).foregroundColor(.white), at: CGPoint(x: r.midX, y: r.midY))
            }
        }
        let range = payload.timeRange
        let axisY = CGFloat(payload.lanes.count) * laneHeight + 9
        for k in 0...4 {
            let t = range.lowerBound + Double(k) / 4 * (range.upperBound - range.lowerBound)
            context.draw(Text(format(t) + " " + payload.unit).font(.caption2).foregroundColor(.secondary),
                         at: CGPoint(x: x(t, width: width), y: axisY))
        }
    }

    private func span(at p: CGPoint, width: CGFloat) -> Int? {
        payload.spans.indices.last { rect(payload.spans[$0], width: width).contains(p) }
    }

    private var readout: String {
        guard let hovered, payload.spans.indices.contains(hovered) else {
            return "\(payload.spans.count) spans on \(payload.lanes.count) lanes · busy "
                + String(format: "%.0f%%", payload.utilization * 100)
        }
        let s = payload.spans[hovered]
        let lane = payload.lanes.indices.contains(s.lane) ? payload.lanes[s.lane] : "?"
        let what = [s.label, s.kind].compactMap { $0 }.joined(separator: ", ")
        return "\(lane): \(what.isEmpty ? "span" : what) from \(format(s.start)) to \(format(s.end)) \(payload.unit)"
            + " (\(format(s.end - s.start)))"
    }
}
