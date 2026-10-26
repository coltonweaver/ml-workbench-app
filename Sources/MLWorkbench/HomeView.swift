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
            VStack(alignment: .leading, spacing: Theme.Space.xxl) {
                CourseHero()
                ModuleGrid()
                WarmupsSection()
                PredictionScoreSection(score: model.review?.score ?? .empty)
            }
            .padding(Theme.Space.xxl)
            .frame(maxWidth: 980, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .softScrollEdges()
        .background(Color(light: NSColor(hex: 0xF6F6F8), dark: NSColor(hex: 0x17181B)))
    }
}

/// The big card at the top: greeting, overall progress and the next lesson.
private struct CourseHero: View {
    @EnvironmentObject private var model: AppModel
    @State private var drift = false

    private var lessons: [LessonSummary] { model.modules.flatMap(\.lessons) }

    /// The first lesson in progress, else the first not started.
    private var next: LessonSummary? {
        lessons.first { $0.status == "in_progress" } ?? lessons.first { $0.status == "new" }
    }

    private var greeting: String {
        let hour = Calendar.current.component(.hour, from: .now)
        let part = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening"
        let first = NSFullUserName().split(separator: " ").first.map(String.init) ?? ""
        return first.isEmpty ? part : "\(part), \(first)"
    }

    var body: some View {
        let passed = lessons.filter { $0.status == "passed" }.count
        let progress = lessons.isEmpty ? 0 : Double(passed) / Double(lessons.count)
        HStack(alignment: .center, spacing: Theme.Space.xl) {
            VStack(alignment: .leading, spacing: 10) {
                Eyebrow("ML Workbench", color: .white.opacity(0.75))
                Text(greeting)
                    .font(Theme.display(34))
                    .foregroundStyle(.white)
                if lessons.isEmpty {
                    Text("Loading lessons…")
                        .foregroundStyle(.white.opacity(0.8))
                } else {
                    Text(passed == 0 ? "From a single derivative to training at scale, one lesson at a time."
                         : "\(passed) of \(lessons.count) lessons passed. Keep the streak going.")
                        .font(.system(size: 14))
                        .foregroundStyle(.white.opacity(0.85))
                        .fixedSize(horizontal: false, vertical: true)
                    if let next {
                        let blocked = model.review?.warmups.isEmpty == false && next.status == "new"
                        Button {
                            model.selection = .lesson(next.id)
                        } label: {
                            Label("\(next.status == "new" ? "Start" : "Continue")  \(next.id) \(next.title)",
                                  systemImage: "arrow.right")
                                .font(.system(size: 13, weight: .semibold))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                        }
                        .secondaryButton()
                        .controlSize(.large)
                        .tint(.white)
                        .disabled(blocked)
                        .help(blocked ? "Do today's warm-ups first" : "")
                        .padding(.top, 6)
                    }
                }
            }
            Spacer(minLength: 0)
            if !lessons.isEmpty {
                ZStack {
                    ProgressRing(progress: progress, lineWidth: 10, tint: .white)
                    VStack(spacing: 0) {
                        Text("\(Int((progress * 100).rounded()))%")
                            .font(Theme.display(26))
                            .contentTransition(.numericText())
                        Text("complete")
                            .font(.caption)
                            .opacity(0.8)
                    }
                    .foregroundStyle(.white)
                }
                .frame(width: 120, height: 120)
            }
        }
        .padding(Theme.Space.xxl)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            MeshGradient(
                width: 3, height: 3,
                points: [
                    [0, 0], [drift ? 0.6 : 0.4, 0], [1, 0],
                    [0, drift ? 0.4 : 0.6], [drift ? 0.35 : 0.65, drift ? 0.6 : 0.4], [1, drift ? 0.6 : 0.4],
                    [0, 1], [drift ? 0.4 : 0.6, 1], [1, 1],
                ],
                colors: [
                    .indigo, .purple, .pink,
                    .blue, .indigo, .purple,
                    .teal, .blue, .indigo,
                ]
            )
            .overlay(Color.black.opacity(0.12))
        }
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.large, style: .continuous))
        .shadow(color: .indigo.opacity(0.25), radius: 24, y: 12)
        .onAppear {
            withAnimation(.easeInOut(duration: 9).repeatForever(autoreverses: true)) { drift = true }
        }
    }
}

/// One card per module with its progress; a click opens where you left off.
private struct ModuleGrid: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        if !model.modules.isEmpty {
            VStack(alignment: .leading, spacing: Theme.Space.m) {
                SectionTitle("Modules", systemImage: "square.grid.2x2.fill", tint: .blue)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 230), spacing: Theme.Space.m)],
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

    private static let tints: [Color] = [.orange, .pink, .purple, .indigo, .blue, .teal, .green, .mint]

    var body: some View {
        let passed = module.lessons.filter { $0.status == "passed" }.count
        let total = max(module.lessons.count, 1)
        let tint = Self.tints[(module.number - 1 + Self.tints.count) % Self.tints.count]
        Button {
            let target = module.lessons.first { $0.status != "passed" } ?? module.lessons.first
            if let target { model.selection = .lesson(target.id) }
        } label: {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("\(module.number)")
                        .font(Theme.display(15))
                        .foregroundStyle(.white)
                        .frame(width: 30, height: 30)
                        .background(tint.gradient, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                    Spacer()
                    if passed == module.lessons.count, passed > 0 {
                        Image(systemName: "checkmark.seal.fill")
                            .foregroundStyle(.white, Theme.pass)
                            .font(.system(size: 18))
                    } else {
                        Text("\(passed)/\(module.lessons.count)")
                            .font(.system(size: 11, weight: .semibold))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
                Text(module.title)
                    .font(.system(size: 14, weight: .semibold))
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                Text(module.summary)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                Spacer(minLength: 0)
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(tint.opacity(0.15))
                        Capsule().fill(tint.gradient)
                            .frame(width: max(4, geo.size.width * Double(passed) / Double(total)))
                    }
                }
                .frame(height: 5)
            }
            .padding(Theme.Space.l)
            .frame(maxWidth: .infinity, minHeight: 168, alignment: .topLeading)
            .card(tint: hovering ? tint : nil)
            .scaleEffect(hovering ? 1.015 : 1)
            .contentShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(Theme.Motion.snappy, value: hovering)
    }
}

/// A section heading with a tinted icon.
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
            IconTile(systemImage: systemImage, tint: tint, size: 24)
            Text(title)
                .font(Theme.display(20))
        }
    }
}

// MARK: Warm-ups

private struct WarmupsSection: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionTitle("Warm-ups", systemImage: "flame.fill", tint: .orange)
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
                    .prominentButton()
                    .tint(.orange)
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
        .padding(Theme.Space.l)
        .card(tint: .orange)
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
        .padding(Theme.Space.l)
        .card(tint: result.correct ? Theme.pass : Theme.warn)
        .transition(.move(edge: .top).combined(with: .opacity))
    }
}

// MARK: Prediction score

private struct PredictionScoreSection: View {
    let score: PredictionScore

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionTitle("Prediction score", systemImage: "scope", tint: .purple)
            Text(score.verdict)
                .foregroundStyle(.secondary)
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
                .font(Theme.display(30, weight: .semibold))
                .contentTransition(.numericText())
            Text(label).font(.caption).foregroundStyle(.secondary)
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
