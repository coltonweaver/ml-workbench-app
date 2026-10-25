import AppKit
import SwiftUI

@main
struct MLWorkbenchApp: App {
    @StateObject private var model = AppModel()

    init() {
        // Launched via `swift run` there's no .app bundle, so ask AppKit to
        // treat us as a regular foreground app with a Dock icon and menus.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate()
    }

    var body: some Scene {
        WindowGroup("ML Workbench") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 900, minHeight: 600)
                .onAppear { model.start() }
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
                    model.stopKernel()
                }
        }
        .commands {
            CommandGroup(replacing: .saveItem) {
                Button("Save") { model.save() }
                    .keyboardShortcut("s")
            }
            CommandMenu("Code") {
                Button("Run") { model.run() }
                    .keyboardShortcut("r")
                    .disabled(!model.canRun)
                Button("Test") { model.test() }
                    .keyboardShortcut("u")
                    .disabled(!model.canRun)
                Button("Check Lesson") { model.check() }
                    .keyboardShortcut("k")
                    .disabled(!model.canCheck)
                Button("Stop") { model.interrupt() }
                    .keyboardShortcut(".")
                Divider()
                Button("Restart Kernel") { model.restartKernel() }
                    .keyboardShortcut("0", modifiers: [.command, .shift])
            }
        }
    }
}
