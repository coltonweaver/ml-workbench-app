import Charts
import SwiftUI
import WorkbenchCore

/// The home screen: where you are in the course, warm-ups for the concepts
/// you've missed more than once, and how well your predictions are calibrated.
struct HomeView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                CourseHeader()
                WarmupsSection()
                PredictionScoreSection(score: model.review?.score ?? .empty)
            }
            .padding(24)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .background(Color(nsColor: .textBackgroundColor))
    }
}

private struct CourseHeader: View {
    @EnvironmentObject private var model: AppModel

    private var lessons: [LessonSummary] { model.modules.flatMap(\.lessons) }

    /// The first lesson in progress, else the first not started.
    private var next: LessonSummary? {
        lessons.first { $0.status == "in_progress" } ?? lessons.first { $0.status == "new" }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("ML Workbench")
                .font(.largeTitle.weight(.semibold))
            if lessons.isEmpty {
                Text("Loading lessons…").foregroundStyle(.secondary)
            } else {
                let passed = lessons.filter { $0.status == "passed" }.count
                ProgressView(value: Double(passed), total: Double(lessons.count)) {
                    Text("\(passed) of \(lessons.count) lessons passed")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: 360)
                if let next {
                    Button {
                        model.selection = .lesson(next.id)
                    } label: {
                        Label("\(next.status == "new" ? "Start" : "Continue") \(next.id): \(next.title)",
                              systemImage: "arrow.right.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.review?.warmups.isEmpty == false && next.status == "new")
                    .help(model.review?.warmups.isEmpty == false && next.status == "new"
                          ? "Do today's warm-ups first" : "")
                }
            }
        }
    }
}

// MARK: Warm-ups

private struct WarmupsSection: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Warm-ups", systemImage: "flame")
                .font(.title2.weight(.semibold))
            let warmups = model.review?.warmups ?? []
            ForEach(model.warmupResults.prefix(3), id: \.id) { result in
                WarmupResultCard(result: result)
            }
            if warmups.isEmpty {
                Text(model.warmupResults.isEmpty
                     ? "Nothing to review. A concept you miss in two separate checks comes back here as a short problem next session."
                     : "That's all for today.")
                    .foregroundStyle(.secondary)
            } else {
                Text("Concepts you've missed more than once. One short problem each.")
                    .foregroundStyle(.secondary)
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
                    .foregroundStyle(.secondary)
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
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(draft.wrappedValue.trimmingCharacters(in: .whitespaces).isEmpty
                              || model.kernelState != .idle)
                if let message = warmup.lastMessage {
                    Text("Last time: \(message)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .help(message)
                }
            }
        }
        .padding(14)
        .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
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
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Revisit \(result.lesson) \(result.lessonTitle)") {
                    model.selection = .lesson(result.lesson)
                }
                .buttonStyle(.link)
            }
        }
        .padding(14)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: Prediction score

private struct PredictionScoreSection: View {
    let score: PredictionScore

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Prediction score", systemImage: "scope")
                .font(.title2.weight(.semibold))
            Text(score.verdict)
                .foregroundStyle(.secondary)
            if score.count > 0 {
                HStack(spacing: 28) {
                    stat(percent(score.hitRate), "right", "\(score.correct) of \(score.count) predictions")
                    if let brier = score.brier {
                        stat(String(format: "%.2f", brier), "Brier score",
                             "0 is perfect; always saying 50% scores 0.25")
                    }
                }
                HStack(alignment: .top, spacing: 24) {
                    if !score.buckets.isEmpty {
                        CalibrationChart(buckets: score.buckets)
                            .frame(width: 260, height: 220)
                    }
                    if score.history.count > 1 {
                        RunningHitRateChart(history: score.history)
                            .frame(minWidth: 260, maxWidth: .infinity, minHeight: 220, maxHeight: 220)
                    }
                }
            }
        }
    }

    private func stat(_ value: String, _ label: String, _ help: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value).font(.system(size: 30, weight: .semibold, design: .rounded))
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
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
                            Text("\(bucket.count)").font(.caption2).foregroundStyle(.secondary)
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
