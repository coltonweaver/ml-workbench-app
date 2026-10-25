import SwiftUI
import WorkbenchCore

/// The lesson flow: hook, predict, build, check, then explain and stretch once
/// the hidden tests pass.
struct LessonPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        if let lesson = model.lesson {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(lesson.moduleTitle.uppercased())
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("\(lesson.id)  \(lesson.title)")
                            .font(.title2.bold())
                    }
                    LessonSection("Problem", systemImage: "questionmark.circle") {
                        MarkdownView(source: lesson.hook)
                    }
                    PredictionCard(lesson: lesson)
                    LessonSection("Build", systemImage: "hammer") {
                        MarkdownView(source: lesson.build)
                        CheckBar(lesson: lesson)
                    }
                    if lesson.isPassed {
                        if let feedback = lesson.progress.feedback {
                            FeedbackCard(feedback: feedback)
                        }
                        if let explain = lesson.explain {
                            LessonSection("Explain", systemImage: "lightbulb") {
                                MarkdownView(source: explain)
                            }
                        }
                        if let stretch = lesson.stretch {
                            LessonSection("Stretch", systemImage: "figure.climbing") {
                                MarkdownView(source: stretch)
                            }
                        }
                    }
                }
                .padding(16)
            }
            .background(Color(nsColor: .windowBackgroundColor))
        }
    }
}

private struct LessonSection<Content: View>: View {
    let title: String
    let systemImage: String
    let content: Content

    init(_ title: String, systemImage: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.systemImage = systemImage
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            content
        }
    }
}

private struct PredictionCard: View {
    @EnvironmentObject private var model: AppModel
    let lesson: LessonDetail

    var body: some View {
        LessonSection("Predict", systemImage: "scope") {
            Text(lesson.predict.prompt)
            if let prediction = lesson.progress.prediction {
                Label("You predicted: \(display(prediction.value))", systemImage: "lock.fill")
                    .foregroundStyle(.secondary)
                if let confidence = prediction.confidence {
                    Text("\(Int((confidence * 100).rounded()))% sure")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                input
                if lesson.predict.kind != "text" {
                    ConfidencePicker(confidence: $model.predictionConfidence)
                }
                HStack {
                    Button("Lock in prediction") { model.submitPrediction() }
                        .keyboardShortcut(.return, modifiers: [.command])
                        .disabled(model.predictionDraft.trimmingCharacters(in: .whitespaces).isEmpty
                                  || model.kernelState != .idle)
                    Text("Run and Check unlock after you predict.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(12)
        .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private var input: some View {
        switch lesson.predict.kind {
        case "choice":
            Picker("", selection: $model.predictionDraft) {
                ForEach(Array((lesson.predict.options ?? []).enumerated()), id: \.offset) { i, option in
                    Text(option).tag(String(i))
                }
            }
            .pickerStyle(.radioGroup)
            .labelsHidden()
        case "number":
            TextField("A number", text: $model.predictionDraft)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 160)
        default:
            TextField("Your prediction", text: $model.predictionDraft, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(2...5)
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

private struct CheckBar: View {
    @EnvironmentObject private var model: AppModel
    let lesson: LessonDetail

    var body: some View {
        HStack(spacing: 10) {
            Button {
                model.check()
            } label: {
                Label("Check", systemImage: "checkmark.seal")
            }
            .disabled(!model.canCheck)
            .help("Run the lesson's hidden tests against \(lesson.file) (⌘K)")
            if lesson.isPassed {
                Label("Passed", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(Color.green)
            } else if let report = model.testReport, model.kernelState != .busy {
                Text(report.outcome == "passed" ? "Passed" : "\(report.failed) of \(report.passed + report.failed) checks failing. See the Tests tab.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if lesson.progress.attempts > 0 {
                Text("\(lesson.progress.attempts) attempt\(lesson.progress.attempts == 1 ? "" : "s") so far")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text(lesson.file)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
        }
        .padding(.top, 4)
    }
}

private struct FeedbackCard: View {
    let feedback: LessonDetail.Feedback

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            switch feedback.correct {
            case true?:
                Label("Your prediction was right", systemImage: "checkmark.circle")
                    .foregroundStyle(Color.green)
            case false?:
                Label("Your prediction was off", systemImage: "xmark.circle")
                    .foregroundStyle(Color.orange)
            case nil:
                Label("Compare your prediction with what you found", systemImage: "text.bubble")
            }
            Text("You said: \(feedback.yours)")
            if let answer = feedback.answer, feedback.correct != true {
                Text("Answer: \(answer)")
            }
        }
        .font(.callout)
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }
}


/// How sure the learner is, for the prediction score on the home screen.
/// Optional: leaving it unset still records the prediction.
struct ConfidencePicker: View {
    @Binding var confidence: Double?

    static let levels: [(Double, String)] = [(0.5, "Guessing"), (0.7, "Fairly sure"), (0.9, "Sure")]

    var body: some View {
        HStack(spacing: 8) {
            Text("How sure?")
                .font(.caption)
                .foregroundStyle(.secondary)
            Picker("How sure?", selection: $confidence) {
                Text("Skip").tag(Double?.none)
                ForEach(Self.levels, id: \.0) { level, label in
                    Text("\(label) (\(Int(level * 100))%)").tag(Double?.some(level))
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
        }
    }
}
