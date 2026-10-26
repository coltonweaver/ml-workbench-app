import AppKit
import SwiftUI
import WorkbenchCore

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.scenePhase) private var scenePhase

    /// Which screen the detail column shows, for the transition between them.
    private var screen: String {
        switch model.selection {
        case .home?: "home"
        case .lesson(let id)?: model.lesson == nil ? "loading" : "lesson-\(id)"
        case .file(let path)?: "file-\(path)"
        case nil: "none"
        }
    }

    var body: some View {
        NavigationSplitView {
            Sidebar()
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 340)
        } detail: {
            ZStack {
                if model.selection == .home {
                    HomeView()
                        .transition(.opacity)
                } else if model.lesson != nil {
                    LessonWorkspace()
                        .transition(.opacity)
                } else {
                    EditorOutputSplit()
                        .transition(.opacity)
                }
            }
            .animation(Theme.Motion.smooth, value: screen)
        }
        .inspector(isPresented: $model.showTutor) {
            TutorPanel()
                .inspectorColumnWidth(min: 300, ideal: 360, max: 520)
        }
        .toolbar { WorkbenchToolbar(model: model) }
        .navigationTitle(title)
        .navigationSubtitle(subtitle)
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { model.reloadIfChangedOnDisk() }
        }
        .alert("Something went wrong", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var title: String {
        if model.selection == .home { return "Home" }
        if let lesson = model.lesson { return "\(lesson.id)  \(lesson.title)" }
        return model.selectedFile ?? "ML Workbench"
    }

    private var subtitle: String {
        guard model.selection != .home, let file = model.selectedFile else { return "" }
        return file + (model.isDirty ? "  ·  Edited" : "")
    }
}

// MARK: - Lesson workspace

/// The lesson on the left, code and output on the right, with a draggable
/// divider between them.
private struct LessonWorkspace: View {
    @AppStorage("lessonPaneWidth") private var lessonWidth: Double = 430
    @State private var dragStart: Double?

    var body: some View {
        GeometryReader { geo in
            let width = clamped(lessonWidth, total: geo.size.width)
            HStack(spacing: 0) {
                LessonPane()
                    .frame(width: width)
                    .clipped()
                SplitHandle(axis: .vertical)
                    .gesture(
                        DragGesture(minimumDistance: 1, coordinateSpace: .global)
                            .onChanged { value in
                                let start = dragStart ?? Double(width)
                                dragStart = start
                                lessonWidth = Double(clamped(start + Double(value.translation.width), total: geo.size.width))
                            }
                            .onEnded { _ in dragStart = nil }
                    )
                EditorOutputSplit()
                    .frame(maxWidth: .infinity)
            }
        }
    }

    private func clamped(_ value: Double, total: CGFloat) -> CGFloat {
        let maxWidth = max(320, Double(total) - 380)
        return CGFloat(min(max(value, 320), maxWidth))
    }
}

/// A hairline you can drag, with a grip that appears on hover.
struct SplitHandle: View {
    enum Axis { case vertical, horizontal }
    let axis: Axis
    @State private var hovering = false

    var body: some View {
        ZStack {
            Rectangle()
                .fill(Color(nsColor: .separatorColor))
                .frame(width: axis == .vertical ? 1 : nil, height: axis == .horizontal ? 1 : nil)
            Capsule()
                .fill(Color.secondary.opacity(hovering ? 0.55 : 0))
                .frame(width: axis == .vertical ? 4 : 36, height: axis == .vertical ? 36 : 4)
        }
        .frame(width: axis == .vertical ? 9 : nil, height: axis == .horizontal ? 9 : nil)
        .frame(maxWidth: axis == .horizontal ? .infinity : nil, maxHeight: axis == .vertical ? .infinity : nil)
        .contentShape(Rectangle())
        .onHover { inside in
            withAnimation(Theme.Motion.snappy) { hovering = inside }
            if inside {
                (axis == .vertical ? NSCursor.resizeLeftRight : NSCursor.resizeUpDown).push()
            } else {
                NSCursor.pop()
            }
        }
    }
}

/// Editor over output with a draggable divider. Not VSplitView: it hosts each
/// pane separately, so the editor ran up under the toolbar and the output pane
/// got the toolbar's height as an empty inset above it.
struct EditorOutputSplit: View {
    @AppStorage("outputPaneHeight") private var outputHeight: Double = 260
    @State private var dragStartHeight: Double?

    private let minEditor: CGFloat = 160
    private let minOutput: CGFloat = 120

    var body: some View {
        GeometryReader { geo in
            VStack(spacing: 0) {
                EditorPane()
                    .frame(maxHeight: .infinity)
                SplitHandle(axis: .horizontal)
                    .gesture(
                        DragGesture(minimumDistance: 1, coordinateSpace: .global)
                            .onChanged { value in
                                let start = dragStartHeight ?? Double(clamped(outputHeight, total: geo.size.height))
                                dragStartHeight = start
                                outputHeight = Double(clamped(start - Double(value.translation.height), total: geo.size.height))
                            }
                            .onEnded { _ in dragStartHeight = nil }
                    )
                OutputPane()
                    .frame(height: clamped(outputHeight, total: geo.size.height))
            }
        }
    }

    private func clamped(_ height: Double, total: CGFloat) -> CGFloat {
        let maxOutput = max(minOutput, total - minEditor - 9)
        return min(max(CGFloat(height), minOutput), maxOutput)
    }
}

// MARK: - Sidebar

private struct Sidebar: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("collapsedModules") private var collapsedRaw = ""
    @AppStorage("filesExpanded") private var filesExpanded = false

    private var collapsed: Set<String> {
        Set(collapsedRaw.split(separator: ",").map(String.init))
    }

    private func expanded(_ id: String) -> Binding<Bool> {
        Binding(
            get: { !collapsed.contains(id) },
            set: { open in
                var set = collapsed
                if open { set.remove(id) } else { set.insert(id) }
                collapsedRaw = set.sorted().joined(separator: ",")
            }
        )
    }

    var body: some View {
        List(selection: $model.selection) {
            HStack(spacing: 10) {
                IconTile(systemImage: "house.fill", tint: .accentColor, size: 22)
                Text("Home")
                    .font(.system(size: 13, weight: .medium))
                Spacer()
                if let count = model.review?.warmups.count, count > 0 {
                    Text("\(count)")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(Theme.warn, in: Capsule())
                        .help("\(count) warm-up\(count == 1 ? "" : "s") waiting")
                }
            }
            .padding(.vertical, 2)
            .tag(SidebarItem.home)

            ForEach(model.modules) { module in
                Section(isExpanded: expanded(module.id)) {
                    ForEach(module.lessons) { lesson in
                        LessonRow(lesson: lesson)
                            .tag(SidebarItem.lesson(lesson.id))
                    }
                } header: {
                    ModuleHeader(module: module)
                }
            }

            Section(isExpanded: $filesExpanded) {
                ForEach(model.files, id: \.self) { path in
                    Label {
                        Text(path)
                            .font(.system(size: 12, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.head)
                    } icon: {
                        Image(systemName: path.contains("test_") ? "checkmark.seal" : "doc.text")
                            .foregroundStyle(.secondary)
                    }
                    .tag(SidebarItem.file(path))
                }
            } header: {
                Text("Files")
            }
        }
        .listStyle(.sidebar)
        .animation(Theme.Motion.smooth, value: model.modules)
        .safeAreaInset(edge: .bottom) {
            WorkspaceFooter()
        }
    }
}

private struct ModuleHeader: View {
    let module: ModuleSummary

    var body: some View {
        let passed = module.lessons.filter { $0.status == "passed" }.count
        HStack(spacing: 8) {
            Text("\(module.number)")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundStyle(.secondary)
                .frame(width: 18, height: 18)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 5, style: .continuous))
            Text(module.title)
                .lineLimit(1)
            Spacer(minLength: 4)
            ProgressRing(progress: module.lessons.isEmpty ? 0 : Double(passed) / Double(module.lessons.count),
                         lineWidth: 2.5, tint: passed == module.lessons.count ? Theme.pass : .accentColor)
                .frame(width: 13, height: 13)
                .help("\(passed) of \(module.lessons.count) passed")
        }
    }
}

private struct LessonRow: View {
    let lesson: LessonSummary

    var body: some View {
        HStack(spacing: 9) {
            StatusGlyph(status: lesson.status)
                .font(.system(size: 14))
                .frame(width: 18)
            Text(lesson.id)
                .font(.system(size: 11.5, weight: .medium, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(minWidth: 24, alignment: .leading)
            Text(lesson.title)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .padding(.vertical, 1)
        .help("\(lesson.id) \(lesson.title)")
    }
}

private struct WorkspaceFooter: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "externaldrive")
                .foregroundStyle(.secondary)
                .font(.caption)
            Text(model.workspace.root.path.replacingOccurrences(of: NSHomeDirectory(), with: "~"))
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.head)
            Spacer()
            Button {
                NSWorkspace.shared.open(model.workspace.root)
            } label: {
                Image(systemName: "folder")
            }
            .help("Show workspace in Finder")
            Button {
                model.refreshFiles()
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .help("Refresh file list")
        }
        .buttonStyle(.borderless)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .glassCapsule()
        .padding(.horizontal, 10)
        .padding(.bottom, 10)
    }
}

// MARK: - Editor and output

private struct EditorPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        if model.selectedFile == nil {
            ContentUnavailableView("No file open", systemImage: "doc.text",
                                   description: Text("Pick a lesson or a file from the sidebar."))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: EditorTheme.background))
        } else {
            CodeEditor(text: $model.editorText) { code, line, column in
                await model.completions(code: code, line: line, column: column)
            }
        }
    }
}

private struct OutputPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: Theme.Space.s) {
                OutputTabs()
                Spacer()
                if model.outputTab == .output, !model.output.isEmpty {
                    Button {
                        model.clearOutput()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    .help("Clear output")
                    .transition(.opacity)
                }
            }
            .padding(.horizontal, Theme.Space.m)
            .padding(.vertical, Theme.Space.s)
            Group {
                switch model.outputTab {
                case .output: OutputLog(chunks: model.output)
                case .tests: TestResults(report: model.testReport, running: model.kernelState == .busy)
                case .instruments: InstrumentsPane()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .transition(.opacity)
            .id(model.outputTab)
        }
        .animation(Theme.Motion.snappy, value: model.outputTab)
        .background(Color(nsColor: .underPageBackgroundColor).opacity(0.6))
    }
}

/// Output · Tests · Instruments, as a capsule switcher with a sliding pill.
private struct OutputTabs: View {
    @EnvironmentObject private var model: AppModel
    @Namespace private var pill

    private struct Tab: Identifiable {
        let id: AppModel.OutputTab
        let title: String
        let icon: String
    }

    private let tabs = [
        Tab(id: .output, title: "Output", icon: "terminal"),
        Tab(id: .tests, title: "Tests", icon: "checkmark.seal"),
        Tab(id: .instruments, title: "Instruments", icon: "waveform.path.ecg"),
    ]

    var body: some View {
        HStack(spacing: 2) {
            ForEach(tabs) { tab in
                let selected = model.outputTab == tab.id
                Button {
                    withAnimation(Theme.Motion.snappy) { model.outputTab = tab.id }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: tab.icon)
                            .font(.system(size: 11, weight: .semibold))
                        Text(tab.title)
                            .font(.system(size: 12, weight: .medium))
                        badge(for: tab.id)
                    }
                    .padding(.horizontal, 11)
                    .padding(.vertical, 5)
                    .foregroundStyle(selected ? Color.primary : Color.secondary)
                    .background {
                        if selected {
                            Capsule()
                                .fill(Color(light: .white, dark: NSColor(white: 1, alpha: 0.14)))
                                .shadow(color: .black.opacity(0.12), radius: 2, y: 1)
                                .matchedGeometryEffect(id: "pill", in: pill)
                        }
                    }
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .background(Color.primary.opacity(0.06), in: Capsule())
    }

    @ViewBuilder
    private func badge(for tab: AppModel.OutputTab) -> some View {
        switch tab {
        case .tests:
            if let report = model.testReport {
                Circle()
                    .fill(report.outcome == "passed" ? Theme.pass : Theme.fail)
                    .frame(width: 6, height: 6)
                    .transition(.scale)
            }
        case .instruments:
            if !model.instruments.isEmpty {
                Text("\(model.instruments.count)")
                    .font(.system(size: 10, weight: .semibold))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        case .output:
            if model.output.contains(where: { $0.style == .error }) {
                Circle().fill(Theme.fail).frame(width: 6, height: 6)
            }
        }
    }
}

private struct OutputLog: View {
    let chunks: [OutputChunk]

    var body: some View {
        if chunks.isEmpty {
            VStack(spacing: 6) {
                Image(systemName: "terminal")
                    .font(.system(size: 22, weight: .light))
                    .foregroundStyle(.tertiary)
                Text("Run your file with ⌘R. Output shows up here.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(chunks) { chunk in
                            row(chunk)
                                .id(chunk.id)
                        }
                    }
                    .padding(.horizontal, Theme.Space.m)
                    .padding(.vertical, Theme.Space.s)
                }
                .onChange(of: chunks) { _, chunks in
                    if let last = chunks.last {
                        withAnimation(Theme.Motion.snappy) { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func row(_ chunk: OutputChunk) -> some View {
        let text = Text(chunk.text.hasSuffix("\n") ? String(chunk.text.dropLast()) : chunk.text)
            .font(.system(size: 12, design: .monospaced))
            .lineSpacing(2)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        switch chunk.style {
        case .error:
            text
                .foregroundStyle(Theme.fail)
                .padding(10)
                .background(Theme.fail.opacity(0.08), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 1.5).fill(Theme.fail).frame(width: 3).padding(.vertical, 6)
                }
                .padding(.vertical, 4)
        case .stderr:
            text.foregroundStyle(Theme.warn)
        case .info:
            text.foregroundStyle(.secondary)
        case .stdout, .result:
            text.foregroundStyle(.primary)
        }
    }
}

private struct TestResults: View {
    let report: TestReport?
    let running: Bool

    var body: some View {
        if let report {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.s) {
                    summary(report)
                    ForEach(Array(report.tests.enumerated()), id: \.element.id) { index, test in
                        TestRow(test: test)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                            .animation(Theme.Motion.smooth.delay(Double(index) * 0.03), value: report.tests.count)
                    }
                    if report.tests.isEmpty || report.outcome == "error" {
                        Text(report.output)
                            .font(.system(size: 11.5, design: .monospaced))
                            .textSelection(.enabled)
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .card(radius: 10)
                    }
                }
                .padding(Theme.Space.m)
            }
        } else if running {
            VStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text("Running tests…").font(.callout).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView("No test run yet", systemImage: "checkmark.seal",
                                   description: Text("Press Test (⌘U) to run the tests for the open file."))
        }
    }

    private func summary(_ r: TestReport) -> some View {
        let passed = r.outcome == "passed"
        let total = max(r.passed + r.failed, 1)
        return HStack(spacing: 10) {
            Image(systemName: passed ? "checkmark.seal.fill" : "xmark.seal.fill")
                .font(.system(size: 20))
                .foregroundStyle(passed ? Theme.pass : Theme.fail)
                .symbolEffect(.bounce, value: r.passed)
            VStack(alignment: .leading, spacing: 1) {
                Text(title(r)).font(.system(size: 13, weight: .semibold))
                if r.passed + r.failed > 0 {
                    Text("\(r.passed) passed · \(r.failed) failed")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            if r.passed + r.failed > 0 {
                ProgressRing(progress: Double(r.passed) / Double(total), lineWidth: 3.5,
                             tint: passed ? Theme.pass : Theme.warn)
                    .frame(width: 22, height: 22)
            }
        }
        .padding(.bottom, 4)
    }

    private func title(_ r: TestReport) -> String {
        switch r.outcome {
        case "passed": "All \(r.passed) passed"
        case "failed": "\(r.failed) failing"
        case "no_tests": "No tests found"
        case "timeout": "Timed out"
        case "cancelled": "Stopped"
        default: "pytest error"
        }
    }
}

private struct TestRow: View {
    let test: TestCase

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: test.didPass ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundStyle(.white, test.didPass ? Theme.pass : Theme.fail)
                Text(displayName)
                    .font(.system(size: 12.5, weight: .medium, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Text(String(format: "%.0f ms", test.duration * 1000))
                    .font(.caption2)
                    .monospacedDigit()
                    .foregroundStyle(.tertiary)
            }
            if let message = test.message, !test.didPass {
                Text(message)
                    .font(.system(size: 11.5, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Theme.fail.opacity(0.06), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            }
        }
        .padding(10)
        .card(tint: test.didPass ? nil : Theme.fail, radius: 10)
        .help(test.nodeid)
    }

    /// `test_slope.py::test_matches_numeric` → `test_matches_numeric`.
    private var displayName: String {
        test.nodeid.components(separatedBy: "::").last ?? test.nodeid
    }
}

// MARK: - Toolbar

private struct WorkbenchToolbar: ToolbarContent {
    @ObservedObject var model: AppModel

    var body: some ToolbarContent {
        ToolbarItem(placement: .status) {
            KernelStatus(state: model.kernelState)
        }
        ToolbarItemGroup(placement: .primaryAction) {
            Button { model.run() } label: { Label("Run", systemImage: "play.fill") }
                .disabled(!model.canRun)
                .help("Run this file in the kernel (⌘R)")
            Button { model.test() } label: { Label("Test", systemImage: "checkmark.seal") }
                .disabled(!model.canRun)
                .help("Run the tests for this file (⌘U)")
            Button { model.interrupt() } label: { Label("Stop", systemImage: "stop.fill") }
                .disabled(model.kernelState != .busy)
                .help("Interrupt (⌘.)")
        }
        ToolbarItem(placement: .primaryAction) {
            Button { withAnimation(Theme.Motion.smooth) { model.showTutor.toggle() } } label: {
                Label("Tutor", systemImage: model.showTutor ? "graduationcap.fill" : "graduationcap")
            }
            .help("Show or hide the tutor (⇧⌘T)")
        }
    }
}

private struct KernelStatus: View {
    let state: KernelState

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "circle.fill")
                .font(.system(size: 7))
                .foregroundStyle(color)
                .symbolEffect(.pulse, options: .repeating, isActive: state == .busy || state == .starting)
            Text(state.label)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(.secondary)
                .contentTransition(.opacity)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .animation(Theme.Motion.snappy, value: state)
        .help(help)
    }

    private var color: Color {
        switch state {
        case .idle: Theme.pass
        case .busy, .starting: .yellow
        case .stopped: .gray
        case .failed: Theme.fail
        }
    }

    private var help: String {
        if case .failed(let reason) = state { return "Kernel failed: \(reason)" }
        return "Python kernel: \(state.label)"
    }
}
