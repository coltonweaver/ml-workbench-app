import AppKit
import SwiftUI
import WorkbenchCore

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationSplitView {
            FileList()
                .navigationSplitViewColumnWidth(min: 180, ideal: 220)
        } detail: {
            VSplitView {
                EditorPane()
                    .frame(minHeight: 200)
                OutputPane()
                    .frame(minHeight: 140, idealHeight: 220)
            }
        }
        .toolbar { WorkbenchToolbar(model: model) }
        .navigationTitle(model.selectedFile.map { $0 + (model.isDirty ? " •" : "") } ?? "ML Workbench")
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
}

private struct FileList: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        List(model.files, id: \.self, selection: $model.selectedFile) { path in
            Label(path, systemImage: path.contains("test_") ? "checkmark.seal" : "doc.text")
                .lineLimit(1)
                .truncationMode(.head)
        }
        .safeAreaInset(edge: .bottom) {
            HStack {
                Text(model.workspace.root.path)
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
                .buttonStyle(.borderless)
                .help("Show workspace in Finder")
                Button {
                    model.refreshFiles()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help("Refresh file list")
            }
            .padding(8)
        }
    }
}

private struct EditorPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        if model.selectedFile == nil {
            ContentUnavailableView("No file open", systemImage: "doc", description: Text("Pick a file from the sidebar."))
        } else {
            CodeEditor(text: $model.editorText)
        }
    }
}

private struct OutputPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $model.outputTab) {
                Text("Output").tag(AppModel.OutputTab.output)
                Text("Tests").tag(AppModel.OutputTab.tests)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 220)
            .padding(6)
            Divider()
            switch model.outputTab {
            case .output: OutputLog(chunks: model.output)
            case .tests: TestResults(report: model.testReport, running: model.kernelState == .busy)
            }
        }
        .background(Color(nsColor: .underPageBackgroundColor))
    }
}

private struct OutputLog: View {
    let chunks: [OutputChunk]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(chunks) { chunk in
                        Text(chunk.text)
                            .font(.system(.callout, design: .monospaced))
                            .foregroundStyle(color(for: chunk.style))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(chunk.id)
                    }
                }
                .padding(8)
            }
            .onChange(of: chunks) { _, chunks in
                if let last = chunks.last { proxy.scrollTo(last.id, anchor: .bottom) }
            }
        }
    }

    private func color(for style: OutputChunk.Style) -> Color {
        switch style {
        case .stdout, .result: .primary
        case .stderr: .orange
        case .error: .red
        case .info: .secondary
        }
    }
}

private struct TestResults: View {
    let report: TestReport?
    let running: Bool

    var body: some View {
        if let report {
            List {
                Section {
                    ForEach(report.tests) { test in
                        VStack(alignment: .leading, spacing: 2) {
                            Label(test.nodeid, systemImage: test.didPass ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundStyle(test.didPass ? Color.green : Color.red)
                            if let message = test.message {
                                Text(message)
                                    .font(.system(.callout, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                        }
                    }
                } header: {
                    Text(summary(report))
                }
                if report.tests.isEmpty || report.outcome == "error" {
                    Section("pytest output") {
                        Text(report.output)
                            .font(.system(.callout, design: .monospaced))
                            .textSelection(.enabled)
                    }
                }
            }
        } else if running {
            ProgressView("Running tests…")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView("No test run yet", systemImage: "checkmark.seal",
                                   description: Text("Press Test (⌘U) to run the tests for the open file."))
        }
    }

    private func summary(_ r: TestReport) -> String {
        switch r.outcome {
        case "passed": "All \(r.passed) passed"
        case "failed": "\(r.failed) failed, \(r.passed) passed"
        case "no_tests": "No tests found"
        case "timeout": "Timed out"
        case "cancelled": "Stopped"
        default: "pytest error"
        }
    }
}

private struct WorkbenchToolbar: ToolbarContent {
    @ObservedObject var model: AppModel

    var body: some ToolbarContent {
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
        ToolbarItem(placement: .status) {
            HStack(spacing: 6) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(model.kernelState.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var statusColor: Color {
        switch model.kernelState {
        case .idle: .green
        case .busy, .starting: .yellow
        case .stopped: .gray
        case .failed: .red
        }
    }
}
