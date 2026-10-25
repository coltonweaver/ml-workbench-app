// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MLWorkbench",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "MLWorkbench", targets: ["MLWorkbench"]),
        .library(name: "WorkbenchCore", targets: ["WorkbenchCore"]),
    ],
    targets: [
        // Everything that isn't UI: the sidecar client, the protocol types,
        // and workspace file handling. Kept free of AppKit so it can be tested
        // headless.
        .target(name: "WorkbenchCore"),
        .executableTarget(
            name: "MLWorkbench",
            dependencies: ["WorkbenchCore"]
        ),
        .testTarget(
            name: "WorkbenchCoreTests",
            dependencies: ["WorkbenchCore"]
        ),
    ],
    swiftLanguageModes: [.v5]
)
