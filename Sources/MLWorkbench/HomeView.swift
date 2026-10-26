import AppKit
import Charts
import SwiftUI
import WorkbenchCore

/// The home screen: where you are in the course, warm-ups for the concepts
/// you've missed more than once, and how well your predictions are calibrated.
struct HomeView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 40) {
                Greeting()
                ModuleGrid()
                WarmupsSection()
                PredictionScoreSection(score: model.review?.score ?? .empty)
            }
            .padding(.horizontal, Theme.Space.xxl)
            .padding(.top, 48)
            .padding(.bottom, Theme.Space.xxl)
            .frame(maxWidth: 900, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .thinScrollBar()
        .background(Theme.Palette.background)
    }
}

/// The greeting, Claude-app style: a serif line with the orange spark,
/// then where you are and what's next.
private struct Greeting: View {
    @EnvironmentObject private var model: AppModel

    private var lessons: [LessonSummary] { model.modules.flatMap(\.lessons) }

    /// The first lesson in progress, else the first not started.
    private var next: LessonSummary? {
        lessons.first { $0.status == "in_progress" } ?? lessons.first { $0.status == "new" }
    }

    private var greeting: String {
        let hour = Calendar.current.component(.hour, from: .now)
        let part = hour < 5 ? "Up late" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening"
        let first = NSFullUserName().split(separator: " ").first.map(String.init) ?? ""
        return first.isEmpty ? part : "\(part), \(first)"
    }

    var body: some View {
        let passed = lessons.filter { $0.status == "passed" }.count
        let progress = lessons.isEmpty ? 0 : Double(passed) / Double(lessons.count)
        VStack(spacing: Theme.Space.l) {
            HStack(spacing: 14) {
                Image(systemName: "asterisk")
                    .font(.system(size: 30, weight: .semibold))
                    .foregroundStyle(Theme.accent)
                Text(greeting)
                    .font(Theme.display(40))
                    .foregroundStyle(Theme.Palette.text)
            }
            if lessons.isEmpty {
                Text("Loading lessons…")
                    .foregroundStyle(Theme.Palette.secondaryText)
            } else {
                VStack(spacing: 8) {
                    Text(passed == 0 ? "From a single derivative to training at scale, one lesson at a time."
                         : "\(passed) of \(lessons.count) lessons passed")
                        .font(.system(size: 14))
                        .foregroundStyle(Theme.Palette.secondaryText)
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Theme.Palette.border)
                            Capsule().fill(Theme.accent)
                                .frame(width: max(4, geo.size.width * progress))
                        }
                    }
                    .frame(width: 280, height: 4)
                    .animation(Theme.Motion.smooth, value: progress)
                }
                if let next {
                    let blocked = model.review?.warmups.isEmpty == false && next.status == "new"
                    Button {
                        model.selection = .lesson(next.id)
                    } label: {
                        HStack(spacing: 8) {
                            Text("\(next.status == "new" ? "Start" : "Continue")  \(next.id) \(next.title)")
                            Image(systemName: "arrow.right")
                                .font(.system(size: 11, weight: .semibold))
                        }
                        .padding(.horizontal, 4)
                        .padding(.vertical, 2)
                    }
                    .prominentButton()
                    .disabled(blocked)
                    .help(blocked ? "Do today's warm-ups first" : "")
                    .padding(.top, 4)
                }
            }
        }
        .frame(maxWidth: .infinity)
    }
}

/// One card per module with its progress; a click opens where you left off.
private struct ModuleGrid: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        if !model.modules.isEmpty {
            VStack(alignment: .leading, spacing: Theme.Space.m) {
                SectionTitle("Modules", systemImage: "square.grid.2x2", tint: Theme.accent)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 250), spacing: Theme.Space.m)],
                          alignment: .leading, spacing: Theme.Space.m) {
                    ForEach(model.modules) { module in
                        ModuleCard(module: module)
                    }
                }
            }
        }
    }
}

private struct ModuleCard: View {
    @EnvironmentObject private var model: AppModel
    let module: ModuleSummary
    @State private var hovering = false

    var body: some View {
        let passed = module.lessons.filter { $0.status == "passed" }.count
        let total = max(module.lessons.count, 1)
        let done = passed == module.lessons.count && passed > 0
        Button {
            let target = module.lessons.first { $0.status != "passed" } ?? module.lessons.first
            if let target { model.selection = .lesson(target.id) }
        } label: {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Module \(module.number)")
                        .font(.system(size: 11.5, weight: .medium))
                        .foregroundStyle(Theme.Palette.secondaryText)
                    Spacer()
                    if done {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(Theme.pass)
                    } else {
                        Text("\(passed)/\(module.lessons.count)")
                            .font(.system(size: 11.5, weight: .medium))
                            .monospacedDigit()
                            .foregroundStyle(Theme.Palette.secondaryText)
                    }
                }
                Text(module.title)
                    .font(Theme.display(18))
                    .foregroundStyle(Theme.Palette.text)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                Text(module.summary)
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.Palette.secondaryText)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                Spacer(minLength: 0)
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Theme.Palette.border)
                        Capsule().fill(done ? Theme.pass : Theme.accent)
                            .frame(width: passed == 0 ? 0 : max(4, geo.size.width * Double(passed) / Double(total)))
                    }
                }
                .frame(height: 3)
            }
            .padding(Theme.Space.l)
            .frame(maxWidth: .infinity, minHeight: 150, alignment: .topLeading)
            .background {
                RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
                    .fill(hovering ? Theme.Palette.hover : Theme.Palette.surface)
            }
            .overlay {
                RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
                    .strokeBorder(hovering ? Theme.accent.opacity(0.5) : Theme.Palette.border, lineWidth: 1)
            }
            .contentShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(Theme.Motion.snappy, value: hovering)
    }
}

/// A section heading in serif with a small line icon.
private struct SectionTitle: View {
    let title: String
    let systemImage: String
    let tint: Color

    init(_ title: String, systemImage: String, tint: Color) {
        self.title = title
        self.systemImage = systemImage
        self.tint = tint
    }

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: systemImage)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(tint)
            Text(title)
                .font(Theme.display(22))
                .foregroundStyle(Theme.Palette.text)
        }
    }
}

// MARK: Warm-ups

private struct WarmupsSection: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionTitle("Warm-ups", systemImage: "flame", tint: Theme.accent)
            let warmups = model.review?.warmups ?? []
            ForEach(model.warmupResults.prefix(3), id: \.id) { result in
                WarmupResultCard(result: result)
            }
            if warmups.isEmpty {
                Text(model.warmupResults.isEmpty
                     ? "Nothing to review. A concept you miss in two separate checks comes back here as a short problem next session."
                     : "That's all for today.")
                    .foregroundStyle(Theme.Palette.secondaryText)
            } else {
                Text("Concepts you've missed more than once. One short problem each.")
                    .foregroundStyle(Theme.Palette.secondaryText)
                ForEach(warmups) { warmup in
                    WarmupCard(warmup: warmup)
                }
            }
        }
    }
}

private struct WarmupCard: View {
    @EnvironmentObject private var model: AppModel
    let warmup: Warmup

    private var draft: Binding<String> {
        Binding(get: { model.warmupDrafts[warmup.id] ?? "" },
                set: { model.warmupDrafts[warmup.id] = $0 })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(warmup.conceptName).font(.headline)
                Spacer()
                Text("missed in \(warmup.mistakes) checks · \(warmup.lesson) \(warmup.lessonTitle)")
                    .font(.caption)
                    .foregroundStyle(Theme.Palette.secondaryText)
            }
            if let context = warmup.context {
                DisclosureGroup("The setup from lesson \(warmup.lesson)") {
                    MarkdownView(source: context)
                        .padding(.top, 4)
                }
            }
            MarkdownView(source: warmup.prompt)
            input
            HStack {
                Button("Answer") { model.answer(warmup) }
                    .prominentButton()
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(draft.wrappedValue.trimmingCharacters(in: .whitespaces).isEmpty
                              || model.kernelState != .idle)
                if let message = warmup.lastMessage {
                    Text("Last time: \(message)")
                        .font(.caption)
                        .foregroundStyle(Theme.Palette.secondaryText)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .help(message)
                }
            }
        }
        .padding(Theme.Space.l)
        .card()
    }

    @ViewBuilder
    private var input: some View {
        if warmup.kind == "choice" {
            Picker("", selection: draft) {
                ForEach(Array((warmup.options ?? []).enumerated()), id: \.offset) { i, option in
                    Text(option).tag(String(i))
                }
            }
            .pickerStyle(.radioGroup)
            .labelsHidden()
        } else {
            TextField("A number", text: draft)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 160)
                .onSubmit { model.answer(warmup) }
        }
    }
}

private struct WarmupResultCard: View {
    @EnvironmentObject private var model: AppModel
    let result: WarmupResult

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(result.correct ? "Right" : "Not quite",
                  systemImage: result.correct ? "checkmark.circle.fill" : "xmark.circle.fill")
                .font(.headline)
                .foregroundStyle(result.correct ? Color.green : Color.orange)
            if !result.correct {
                Text("You said \(result.yours); the answer is \(result.answer).")
            }
            if let explain = result.explain {
                MarkdownView(source: explain)
            }
            HStack {
                Text(result.correct
                     ? "Cleared until you miss it again."
                     : "It stays in your queue with a different problem next time.")
                    .font(.caption)
                    .foregroundStyle(Theme.Palette.secondaryText)
                Spacer()
                Button("Revisit \(result.lesson) \(result.lessonTitle)") {
                    model.selection = .lesson(result.lesson)
                }
                .buttonStyle(.link)
            }
        }
        .padding(Theme.Space.l)
        .card()
        .transition(.move(edge: .top).combined(with: .opacity))
    }
}

// MARK: Prediction score

private struct PredictionScoreSection: View {
    let score: PredictionScore

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionTitle("Prediction score", systemImage: "scope", tint: Theme.accent)
            Text(score.verdict)
                .foregroundStyle(Theme.Palette.secondaryText)
            if score.count > 0 {
                HStack(spacing: Theme.Space.m) {
                    stat(percent(score.hitRate), "right", "\(score.correct) of \(score.count) predictions")
                    if let brier = score.brier {
                        stat(String(format: "%.2f", brier), "Brier score",
                             "0 is perfect; always saying 50% scores 0.25")
                    }
                }
                HStack(alignment: .top, spacing: Theme.Space.m) {
                    if !score.buckets.isEmpty {
                        CalibrationChart(buckets: score.buckets)
                            .frame(width: 260, height: 220)
                            .padding(Theme.Space.l)
                            .card()
                    }
                    if score.history.count > 1 {
                        RunningHitRateChart(history: score.history)
                            .frame(minWidth: 260, maxWidth: .infinity, minHeight: 220, maxHeight: 220)
                            .padding(Theme.Space.l)
                            .card()
                    }
                }
            }
        }
    }

    private func stat(_ value: String, _ label: String, _ help: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(Theme.display(30))
                .contentTransition(.numericText())
            Text(label).font(.caption).foregroundStyle(Theme.Palette.secondaryText)
        }
        .padding(.horizontal, Theme.Space.l)
        .padding(.vertical, Theme.Space.m)
        .frame(minWidth: 140, alignment: .leading)
        .card()
        .help(help)
    }
}

private func percent(_ x: Double?) -> String {
    guard let x else { return "–" }
    return "\(Int((x * 100).rounded()))%"
}

/// Stated confidence against how often you were right at that confidence.
/// On the diagonal is perfectly calibrated; below it is overconfident.
private struct CalibrationChart: View {
    let buckets: [PredictionScore.Bucket]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Calibration").font(.subheadline.weight(.semibold))
            Chart {
                LineMark(x: .value("Said", 0.0), y: .value("Right", 0.0), series: .value("", "ideal"))
                    .foregroundStyle(.secondary.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                LineMark(x: .value("Said", 1.0), y: .value("Right", 1.0), series: .value("", "ideal"))
                    .foregroundStyle(.secondary.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                ForEach(buckets) { bucket in
                    PointMark(x: .value("Said", bucket.confidence), y: .value("Right", bucket.hitRate))
                        .symbolSize(40 + 30 * Double(min(bucket.count, 10)))
                        .annotation(position: .top) {
                            Text("\(bucket.count)").font(.caption2).foregroundStyle(Theme.Palette.secondaryText)
                        }
                }
            }
            .chartXScale(domain: 0...1)
            .chartYScale(domain: 0...1)
            .chartXAxisLabel("How sure you said you were")
            .chartYAxisLabel("How often you were right")
            .chartXAxis { AxisMarks(values: [0.0, 0.5, 0.7, 0.9, 1.0]) { AxisGridLine(); AxisValueLabel(format: FloatingPointFormatStyle<Double>.Percent()) } }
            .chartYAxis { AxisMarks(values: [0.0, 0.25, 0.5, 0.75, 1.0]) { AxisGridLine(); AxisValueLabel(format: FloatingPointFormatStyle<Double>.Percent()) } }
        }
    }
}

/// Hit rate so far after each graded prediction, in the order you made them.
private struct RunningHitRateChart: View {
    let history: [PredictionScore.Point]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Over time").font(.subheadline.weight(.semibold))
            Chart {
                ForEach(Array(history.enumerated()), id: \.offset) { i, point in
                    LineMark(x: .value("Prediction", i + 1), y: .value("Right so far", point.hitRate))
                    PointMark(x: .value("Prediction", i + 1), y: .value("Right so far", point.hitRate))
                        .foregroundStyle(point.correct ? Color.green : Color.orange)
                        .symbolSize(24)
                }
            }
            .chartYScale(domain: 0...1)
            .chartXAxisLabel("Predictions, oldest first")
            .chartYAxis { AxisMarks(values: [0.0, 0.5, 1.0]) { AxisGridLine(); AxisValueLabel(format: FloatingPointFormatStyle<Double>.Percent()) } }
        }
    }
}
