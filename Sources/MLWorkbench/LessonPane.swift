import AppKit
import SwiftUI
import WorkbenchCore

/// The lesson flow: hook, predict, build, check, then explain and stretch once
/// the hidden tests pass.
struct LessonPane: View {
    @EnvironmentObject private var model: AppModel
    @State private var confetti = 0

    private struct PassKey: Equatable {
        let id: String
        let passed: Bool
    }

    var body: some View {
        if let lesson = model.lesson {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.l) {
                    LessonHeader(lesson: lesson)
                    StageTrack(lesson: lesson)
                    StageCard("Problem", systemImage: "questionmark.circle.fill", tint: Theme.Stage.problem) {
                        MarkdownView(source: lesson.hook)
                    }
                    PredictionCard(lesson: lesson)
                    StageCard("Build", systemImage: "hammer.fill", tint: Theme.Stage.build) {
                        MarkdownView(source: lesson.build)
                        CheckBar(lesson: lesson)
                    }
                    .opacity(lesson.hasPrediction ? 1 : 0.55)
                    .saturation(lesson.hasPrediction ? 1 : 0.4)
                    if lesson.isPassed {
                        PassedCard(feedback: lesson.progress.feedback)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                        if let explain = lesson.explain {
                            StageCard("Explain", systemImage: "lightbulb.fill", tint: Theme.Stage.explain, tinted: true) {
                                MarkdownView(source: explain)
                            }
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                        }
                        if let stretch = lesson.stretch {
                            StageCard("Stretch", systemImage: "figure.climbing", tint: Theme.Stage.stretch, tinted: true) {
                                MarkdownView(source: stretch)
                            }
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                        }
                    }
                }
                .padding(.horizontal, Theme.Space.xl)
                .padding(.top, Theme.Space.l)
                .padding(.bottom, Theme.Space.xxl)
                .frame(maxWidth: .infinity, alignment: .leading)
                .animation(Theme.Motion.smooth, value: lesson.isPassed)
                .animation(Theme.Motion.smooth, value: lesson.hasPrediction)
            }
            .thinScrollBar()
            .background(LessonBackground())
            .overlay { ConfettiBurst(trigger: confetti).ignoresSafeArea() }
            .onChange(of: PassKey(id: lesson.id, passed: lesson.isPassed)) { old, new in
                if old.id == new.id, !old.passed, new.passed { confetti += 1 }
            }
        }
    }
}

private struct LessonBackground: View {
    var body: some View {
        Theme.Palette.background
    }
}

private struct LessonHeader: View {
    let lesson: LessonDetail

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Eyebrow("Lesson \(lesson.id) · \(lesson.moduleTitle)", color: Theme.accent)
                .lineLimit(1)
            Text(lesson.title)
                .font(Theme.display(28))
                .foregroundStyle(Theme.Palette.text)
                .fixedSize(horizontal: false, vertical: true)
            if !lesson.concepts.isEmpty {
                FlowLayout(spacing: 6) {
                    ForEach(lesson.concepts, id: \.self) { concept in
                        Text(concept.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: "-", with: " "))
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(Theme.Palette.secondaryText)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.primary.opacity(0.06), in: Capsule())
                    }
                }
                .padding(.top, 2)
            }
        }
    }
}

// MARK: - Stage track

/// Read → Predict → Build → Check → Reflect, filled in as you go.
private struct StageTrack: View {
    let lesson: LessonDetail

    private struct Stage {
        let title: String
        let icon: String
        let tint: Color
        let done: Bool
    }

    private var stages: [Stage] {
        [
            Stage(title: "Read", icon: "book.fill", tint: Theme.Stage.problem, done: true),
            Stage(title: "Predict", icon: "scope", tint: Theme.Stage.predict, done: lesson.hasPrediction),
            Stage(title: "Build", icon: "hammer.fill", tint: Theme.Stage.build,
                  done: lesson.progress.attempts > 0 || lesson.isPassed),
            Stage(title: "Check", icon: "checkmark.seal.fill", tint: Theme.Stage.check, done: lesson.isPassed),
            Stage(title: "Reflect", icon: "lightbulb.fill", tint: Theme.Stage.explain, done: lesson.isPassed),
        ]
    }

    var body: some View {
        let stages = self.stages
        let current = stages.firstIndex { !$0.done }
        HStack(spacing: 0) {
            ForEach(Array(stages.enumerated()), id: \.offset) { i, stage in
                if i > 0 {
                    Capsule()
                        .fill(stages[i - 1].done ? AnyShapeStyle(stages[i - 1].tint) : AnyShapeStyle(Color.primary.opacity(0.1)))
                        .frame(height: 2)
                        .frame(maxWidth: .infinity)
                        .padding(.bottom, 16)
                }
                VStack(spacing: 4) {
                    ZStack {
                        Circle()
                            .fill(stage.done ? AnyShapeStyle(stage.tint) : AnyShapeStyle(Color.primary.opacity(0.06)))
                        if i == current {
                            Circle().strokeBorder(stage.tint, lineWidth: 1.5)
                        }
                        Image(systemName: stage.icon)
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(stage.done ? Color.white : (i == current ? stage.tint : Color.secondary))
                            .symbolEffect(.bounce, value: stage.done)
                    }
                    .frame(width: 24, height: 24)
                    .scaleEffect(i == current ? 1.08 : 1)
                    Text(stage.title)
                        .font(.system(size: 10, weight: i == current ? .semibold : .regular))
                        .foregroundStyle(i == current ? Color.primary : Color.secondary)
                        .fixedSize()
                }
            }
        }
        .animation(Theme.Motion.bouncy, value: current)
        .padding(.vertical, 2)
    }
}

// MARK: - Cards

private struct StageCard<Content: View>: View {
    let title: String
    let systemImage: String
    let tint: Color
    let tinted: Bool
    let content: Content

    init(_ title: String, systemImage: String, tint: Color, tinted: Bool = false,
         @ViewBuilder content: () -> Content) {
        self.title = title
        self.systemImage = systemImage
        self.tint = tint
        self.tinted = tinted
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.m) {
            HStack(spacing: 8) {
                IconTile(systemImage: systemImage, tint: tint, size: 22)
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                Spacer()
            }
            content
        }
        .padding(Theme.Space.l)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card(tint: tinted ? tint : nil)
    }
}

private struct PredictionCard: View {
    @EnvironmentObject private var model: AppModel
    let lesson: LessonDetail

    var body: some View {
        StageCard("Predict", systemImage: "scope", tint: Theme.Stage.predict, tinted: !lesson.hasPrediction) {
            MarkdownView(source: lesson.predict.prompt)
            if let prediction = lesson.progress.prediction {
                HStack(spacing: 10) {
                    Image(systemName: "lock.fill")
                        .foregroundStyle(Theme.Stage.predict)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Locked in")
                            .font(.caption)
                            .foregroundStyle(Theme.Palette.secondaryText)
                        Text(display(prediction.value))
                            .font(.system(size: 15, weight: .semibold, design: .rounded))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer()
                    if let confidence = prediction.confidence {
                        Text("\(Int((confidence * 100).rounded()))% sure")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(Theme.Stage.predict)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Theme.Stage.predict.opacity(0.12), in: Capsule())
                    }
                }
                .padding(Theme.Space.m)
                .background(Theme.Stage.predict.opacity(0.07), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                .transition(.scale(scale: 0.96).combined(with: .opacity))
            } else {
                VStack(alignment: .leading, spacing: Theme.Space.m) {
                    input
                    if lesson.predict.kind != "text" {
                        ConfidencePicker(confidence: $model.predictionConfidence)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        lockButton
                        hint
                    }
                }
                .transition(.opacity)
            }
        }
    }

    private var lockButton: some View {
        Button {
            withAnimation(Theme.Motion.smooth) { model.submitPrediction() }
        } label: {
            Label("Lock in prediction", systemImage: "lock.fill")
                .padding(.horizontal, 4)
        }
        .prominentButton()
        .controlSize(.large)
        .keyboardShortcut(.return, modifiers: [.command])
        .disabled(model.predictionDraft.trimmingCharacters(in: .whitespaces).isEmpty
                  || model.kernelState != .idle)
    }

    private var hint: some View {
        Text("Run and Check unlock once you've predicted.")
            .font(.caption)
            .foregroundStyle(Theme.Palette.secondaryText)
            .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder
    private var input: some View {
        switch lesson.predict.kind {
        case "choice":
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array((lesson.predict.options ?? []).enumerated()), id: \.offset) { i, option in
                    ChoiceRow(text: option, selected: model.predictionDraft == String(i)) {
                        withAnimation(Theme.Motion.snappy) { model.predictionDraft = String(i) }
                    }
                }
            }
        case "number":
            FieldBox {
                TextField("A number", text: $model.predictionDraft)
                    .textFieldStyle(.plain)
                    .font(.system(size: 15, weight: .medium, design: .rounded))
                    .monospacedDigit()
            }
            .frame(maxWidth: 200)
        default:
            FieldBox {
                TextField("Your prediction", text: $model.predictionDraft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(2...6)
            }
        }
    }

    private func display(_ value: String) -> String {
        if lesson.predict.kind == "choice", let i = Int(value), let options = lesson.predict.options,
           options.indices.contains(i) {
            return options[i]
        }
        return value
    }
}

/// A text field in a soft rounded box that lights up when focused.
private struct FieldBox<Content: View>: View {
    @ViewBuilder var content: Content
    @FocusState private var focused: Bool

    var body: some View {
        content
            .focused($focused)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(Theme.Palette.background, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(focused ? Theme.accent : Theme.Palette.border, lineWidth: focused ? 1.5 : 1)
            }
            .animation(Theme.Motion.snappy, value: focused)
    }
}

private struct ChoiceRow: View {
    let text: String
    let selected: Bool
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(selected ? Theme.accent : Theme.Palette.secondaryText)
                    .contentTransition(.symbolEffect(.replace))
                Text(text)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .fill(selected ? Theme.accent.opacity(0.10) : hovering ? Theme.Palette.hover : Theme.Palette.background)
            }
            .overlay {
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(selected ? Theme.accent.opacity(0.7) : Theme.Palette.border, lineWidth: 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(Theme.Motion.snappy, value: hovering)
    }
}

/// How sure the learner is, for the prediction score on the home screen.
/// Optional: leaving it unset still records the prediction.
struct ConfidencePicker: View {
    @Binding var confidence: Double?

    static let levels: [(Double, String)] = [(0.5, "Guessing"), (0.7, "Fairly sure"), (0.9, "Sure")]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("How sure are you?")
                .font(.caption)
                .foregroundStyle(Theme.Palette.secondaryText)
            FlowLayout(spacing: 6) {
                ForEach(Self.levels, id: \.0) { level, label in
                    let selected = confidence == level
                    Button {
                        withAnimation(Theme.Motion.snappy) { confidence = selected ? nil : level }
                    } label: {
                        HStack(spacing: 4) {
                            Text(label)
                            Text("\(Int(level * 100))%")
                                .foregroundStyle(selected ? Color.white.opacity(0.8) : Color.secondary)
                                .monospacedDigit()
                        }
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(selected ? Color.white : Theme.Palette.text)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 5)
                        .background {
                            Capsule().fill(selected ? AnyShapeStyle(Theme.accent)
                                                    : AnyShapeStyle(Theme.Palette.hover))
                        }
                        .contentShape(Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

private struct CheckBar: View {
    @EnvironmentObject private var model: AppModel
    let lesson: LessonDetail

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider().opacity(0.5)
            HStack(spacing: 10) {
                Button {
                    model.check()
                } label: {
                    Label(lesson.isPassed ? "Check again" : "Check", systemImage: "checkmark.seal.fill")
                        .padding(.horizontal, 2)
                }
                .prominentButton()
                .controlSize(.large)
                .disabled(!model.canCheck)
                .help("Run the lesson's hidden tests against \(lesson.file) (⌘K)")
                status
                    .transition(.opacity)
                Spacer(minLength: 0)
            }
            .animation(Theme.Motion.snappy, value: model.kernelState)
            Label(lesson.file, systemImage: "doc.text")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Theme.Palette.secondaryText)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(.top, 2)
    }

    @ViewBuilder
    private var status: some View {
        if model.kernelState == .busy {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Checking…").font(.caption).foregroundStyle(Theme.Palette.secondaryText)
            }
        } else if lesson.isPassed {
            Label("Passed", systemImage: "checkmark.circle.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Theme.pass)
        } else if let report = model.testReport {
            Text(report.outcome == "passed" ? "Passed"
                 : "\(report.failed) of \(report.passed + report.failed) checks failing. See Tests.")
                .font(.caption)
                .foregroundStyle(Theme.Palette.secondaryText)
                .fixedSize(horizontal: false, vertical: true)
        } else if lesson.progress.attempts > 0 {
            Text("\(lesson.progress.attempts) attempt\(lesson.progress.attempts == 1 ? "" : "s") so far")
                .font(.caption)
                .foregroundStyle(Theme.Palette.secondaryText)
        }
    }
}

/// The moment the hidden tests pass, and how the prediction held up.
private struct PassedCard: View {
    let feedback: LessonDetail.Feedback?
    @State private var appeared = false

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Space.m) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 30))
                .foregroundStyle(.white, Theme.pass)
                .symbolEffect(.bounce, value: appeared)
            VStack(alignment: .leading, spacing: 4) {
                Text("Lesson passed")
                    .font(Theme.display(19))
                if let feedback {
                    switch feedback.correct {
                    case true?:
                        Label("Your prediction was right", systemImage: "scope")
                            .foregroundStyle(Theme.pass)
                    case false?:
                        Label("Your prediction was off", systemImage: "scope")
                            .foregroundStyle(Theme.warn)
                    case nil:
                        Label("Compare your prediction with what you found", systemImage: "text.bubble")
                            .foregroundStyle(Theme.Palette.secondaryText)
                    }
                    Text(summary(feedback))
                        .foregroundStyle(Theme.Palette.secondaryText)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .font(.callout)
            Spacer(minLength: 0)
        }
        .padding(Theme.Space.l)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card(tint: Theme.pass)
        .onAppear { appeared = true }
    }

    private func summary(_ feedback: LessonDetail.Feedback) -> String {
        var text = "You said \(feedback.yours)."
        if feedback.correct != true, let answer = feedback.answer {
            text += " The answer: \(answer)."
        }
        return text
    }
}

// MARK: - Confetti

/// A short burst of confetti, fired each time `trigger` changes.
struct ConfettiBurst: View {
    let trigger: Int
    @State private var start: Date?

    private static let duration: TimeInterval = 2.6
    private static let colors: [Color] = [.pink, .orange, .yellow, .green, .teal, .blue, .purple]

    private struct Piece {
        let x: Double, vx: Double, vy: Double, spin: Double, size: Double, color: Int, delay: Double
    }

    /// Deterministic, so every burst looks alike and nothing is allocated per frame.
    private static let pieces: [Piece] = {
        var seed: UInt64 = 0x9E3779B97F4A7C15
        func rand() -> Double {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Double(seed >> 11) / Double(1 << 53)
        }
        return (0..<90).map { _ in
            Piece(x: 0.2 + rand() * 0.6, vx: (rand() - 0.5) * 0.7, vy: -(0.55 + rand() * 0.75),
                  spin: (rand() - 0.5) * 14, size: 5 + rand() * 5, color: Int(rand() * 7) % 7,
                  delay: rand() * 0.15)
        }
    }()

    var body: some View {
        TimelineView(.animation(paused: start == nil)) { context in
            Canvas { ctx, size in
                guard let start else { return }
                let elapsed = context.date.timeIntervalSince(start)
                for p in Self.pieces {
                    let t = elapsed - p.delay
                    guard t > 0 else { continue }
                    let gravity = 1.6
                    let x = (p.x + p.vx * t) * size.width
                    let y = (0.28 + p.vy * t + 0.5 * gravity * t * t) * size.height
                    let fade = max(0, min(1, (Self.duration - elapsed) / 0.6))
                    var piece = ctx
                    piece.opacity = fade
                    piece.translateBy(x: x, y: y)
                    piece.rotate(by: .radians(p.spin * t))
                    let w = p.size, h = p.size * 0.45
                    piece.fill(Path(roundedRect: CGRect(x: -w / 2, y: -h / 2, width: w, height: h), cornerRadius: 1),
                               with: .color(Self.colors[p.color]))
                }
            }
        }
        .allowsHitTesting(false)
        .onChange(of: trigger) { _, _ in
            start = .now
            DispatchQueue.main.asyncAfter(deadline: .now() + Self.duration) { start = nil }
        }
    }
}
