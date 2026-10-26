import AppKit
import SwiftUI

/// Spacing, radii, type and color in one place, so every screen shares them.
/// The look follows the Claude app: warm neutrals, flat surfaces with
/// hairline borders, serif display type and one orange accent.
enum Theme {
    enum Space {
        static let xs: CGFloat = 4
        static let s: CGFloat = 8
        static let m: CGFloat = 12
        static let l: CGFloat = 16
        static let xl: CGFloat = 24
        static let xxl: CGFloat = 32
    }

    enum Radius {
        static let small: CGFloat = 8
        static let card: CGFloat = 14
        static let large: CGFloat = 20
    }

    enum Motion {
        static let snappy = Animation.snappy(duration: 0.28)
        static let smooth = Animation.smooth(duration: 0.4)
        static let bouncy = Animation.bouncy(duration: 0.5, extraBounce: 0.1)
    }

    /// Surfaces and text, light and dark.
    enum Palette {
        /// Main content background.
        static let background = Color(nsColor: .dynamic(light: 0xFAF9F5, dark: 0x262624))
        /// The sidebar, a step darker than content.
        static let sidebar = Color(nsColor: .dynamic(light: 0xF3F1EA, dark: 0x1F1E1D))
        /// Cards and inputs.
        static let surface = Color(nsColor: .dynamic(light: 0xFFFFFF, dark: 0x30302E))
        /// Code and console wells.
        static let well = Color(nsColor: .dynamic(light: 0xF5F4EF, dark: 0x1F1E1D))
        static let border = Color(nsColor: .dynamic(light: 0xE4E1D7, dark: 0x3D3D3A))
        static let text = Color(nsColor: .dynamic(light: 0x141413, dark: 0xF5F4EE))
        static let secondaryText = Color(nsColor: .dynamic(light: 0x73726C, dark: 0xA6A39A))
        /// Selected sidebar row.
        static let selection = Color(nsColor: .dynamic(light: 0xE8E5DA, dark: 0x3A3A37))
        /// Hovered rows and quiet fills.
        static let hover = Color(nsColor: .dynamic(light: 0xEEEBE2, dark: 0x2E2E2C))
    }

    /// Claude's orange.
    static let accent = Color(nsColor: .dynamic(light: 0xC6613F, dark: 0xD97757))

    /// Lesson stages get a muted hue each, used sparingly for icons.
    enum Stage {
        static let problem = Color(nsColor: .dynamic(light: 0x4F7FB0, dark: 0x6A9BCC))
        static let predict = Color(nsColor: .dynamic(light: 0x8A6BBE, dark: 0xA88BD6))
        static let build = accent
        static let check = Color(nsColor: .dynamic(light: 0x5F7F47, dark: 0x8FAA70))
        static let explain = Color(nsColor: .dynamic(light: 0xA9832F, dark: 0xD4B06A))
        static let stretch = Color(nsColor: .dynamic(light: 0x3F8A82, dark: 0x6BB3A9))
    }

    static let pass = Color(nsColor: .dynamic(light: 0x4E8A3E, dark: 0x7DB46A))
    static let warn = Color(nsColor: .dynamic(light: 0xB7791F, dark: 0xE0A84E))
    static let fail = Color(nsColor: .dynamic(light: 0xC2463B, dark: 0xE5675B))

    /// Reading text in the lesson pane.
    static let body = Font.system(size: 14)
    static let bodyLineSpacing: CGFloat = 3.5
    /// Headings in serif, like the Claude app's greeting.
    static func display(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .serif)
    }
    static let eyebrow = Font.system(size: 11, weight: .medium)
    static let mono = Font.system(size: 12.5, design: .monospaced)
}

extension Color {
    /// A light/dark pair as one dynamic color.
    init(light: NSColor, dark: NSColor) {
        self.init(nsColor: NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.darkAqua, .vibrantDark]) != nil ? dark : light
        })
    }
}

extension NSColor {
    convenience init(hex: UInt32, alpha: CGFloat = 1) {
        self.init(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
                  green: CGFloat((hex >> 8) & 0xFF) / 255,
                  blue: CGFloat(hex & 0xFF) / 255,
                  alpha: alpha)
    }

    static func dynamic(light: UInt32, dark: UInt32) -> NSColor {
        NSColor(name: nil) { appearance in
            NSColor(hex: appearance.bestMatch(from: [.darkAqua, .vibrantDark]) != nil ? dark : light)
        }
    }
}

// MARK: Liquid Glass, with a material fallback

/// Glass on macOS 26 (when built with its SDK), frosted material before that.
struct GlassSurface<S: Shape>: ViewModifier {
    var shape: S
    var tint: Color?
    var interactive = false

    func body(content: Content) -> some View {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            content.glassEffect(glass, in: shape)
        } else {
            fallback(content)
        }
        #else
        fallback(content)
        #endif
    }

    #if compiler(>=6.2)
    @available(macOS 26.0, *)
    private var glass: Glass {
        var glass = Glass.regular
        if let tint { glass = glass.tint(tint.opacity(0.22)) }
        if interactive { glass = glass.interactive() }
        return glass
    }
    #endif

    private func fallback(_ content: Content) -> some View {
        content
            .background {
                ZStack {
                    shape.fill(.regularMaterial)
                    if let tint { shape.fill(tint.opacity(0.10)) }
                }
            }
            .overlay { shape.stroke(Color.white.opacity(0.12), lineWidth: 0.5) }
            .shadow(color: .black.opacity(0.08), radius: 10, y: 4)
    }
}

/// A flat card: surface fill and a hairline border, no shadow. A tint
/// washes the fill very lightly.
struct CardSurface: ViewModifier {
    var tint: Color?
    var radius: CGFloat = Theme.Radius.card

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
        content
            .background {
                ZStack {
                    shape.fill(Theme.Palette.surface)
                    if let tint { shape.fill(tint.opacity(0.07)) }
                }
            }
            .overlay { shape.strokeBorder(Theme.Palette.border, lineWidth: 1) }
    }
}

/// The Claude app's filled button: accent background, white label.
struct FilledButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        StyledButton(configuration: configuration, filled: true)
    }
}

/// The Claude app's quiet button: surface fill with a hairline border.
struct OutlineButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        StyledButton(configuration: configuration, filled: false)
    }
}

private struct StyledButton: View {
    let configuration: ButtonStyleConfiguration
    let filled: Bool
    @Environment(\.isEnabled) private var isEnabled
    @State private var hovering = false

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: 9, style: .continuous)
        configuration.label
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(filled ? Color.white : Theme.Palette.text)
            .padding(.horizontal, 14)
            .padding(.vertical, 7)
            .background {
                if filled {
                    shape.fill(.tint)
                        .brightness(configuration.isPressed ? -0.08 : hovering ? 0.05 : 0)
                } else {
                    shape.fill(hovering || configuration.isPressed ? Theme.Palette.hover : Theme.Palette.surface)
                }
            }
            .overlay {
                if !filled { shape.strokeBorder(Theme.Palette.border, lineWidth: 1) }
            }
            .contentShape(shape)
            .opacity(isEnabled ? 1 : 0.45)
            .onHover { hovering = $0 }
            .animation(Theme.Motion.snappy, value: configuration.isPressed)
    }
}

/// Keeps scroll bars thin and floating, as in the Claude app, even when the
/// system shows legacy scroll bars (a mouse is attached). Put it in the
/// background of a ScrollView's content.
struct OverlayScrollers: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView { Probe() }
    func updateNSView(_ view: NSView, context: Context) {}

    final class Probe: NSView {
        private var observer: NSObjectProtocol?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            apply()
            if observer == nil {
                observer = NotificationCenter.default.addObserver(
                    forName: NSScroller.preferredScrollerStyleDidChangeNotification, object: nil, queue: .main
                ) { [weak self] _ in self?.apply() }
            }
        }

        deinit {
            if let observer { NotificationCenter.default.removeObserver(observer) }
        }

        private func apply() {
            // After SwiftUI has finished setting the scroll view up.
            DispatchQueue.main.async { [weak self] in
                guard let scroll = self?.enclosingScrollView else { return }
                scroll.scrollerStyle = .overlay
                scroll.autohidesScrollers = true
            }
        }
    }
}

extension View {
    func glassSurface<S: Shape>(_ shape: S, tint: Color? = nil, interactive: Bool = false) -> some View {
        modifier(GlassSurface(shape: shape, tint: tint, interactive: interactive))
    }

    func glassCapsule(tint: Color? = nil, interactive: Bool = false) -> some View {
        glassSurface(Capsule(), tint: tint, interactive: interactive)
    }

    func card(tint: Color? = nil, radius: CGFloat = Theme.Radius.card) -> some View {
        modifier(CardSurface(tint: tint, radius: radius))
    }

    /// The main action on a screen.
    func prominentButton() -> some View {
        buttonStyle(FilledButtonStyle())
    }

    /// A secondary action.
    func secondaryButton() -> some View {
        buttonStyle(OutlineButtonStyle())
    }

    /// Thin, floating scroll bars for the enclosing ScrollView. Apply to the
    /// scroll view's content.
    func overlayScrollers() -> some View {
        background(OverlayScrollers())
    }

    /// Groups glass shapes so they blend and morph together on macOS 26.
    @ViewBuilder
    func glassGroup(spacing: CGFloat = 8) -> some View {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            GlassEffectContainer(spacing: spacing) { self }
        } else {
            self
        }
        #else
        self
        #endif
    }

    /// Soft fade where content scrolls under a toolbar, on macOS 26.
    @ViewBuilder
    func softScrollEdges() -> some View {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            scrollEdgeEffectStyle(.soft, for: .all)
        } else {
            self
        }
        #else
        self
        #endif
    }
}

// MARK: Small shared pieces

/// A small label above a title.
struct Eyebrow: View {
    let text: String
    var color: Color = .secondary

    init(_ text: String, color: Color = .secondary) {
        self.text = text
        self.color = color
    }

    var body: some View {
        Text(text)
            .font(Theme.eyebrow)
            .foregroundStyle(color)
    }
}

/// A circular progress ring.
struct ProgressRing: View {
    var progress: Double
    var lineWidth: CGFloat = 4
    var tint: Color = Theme.accent

    var body: some View {
        ZStack {
            Circle().stroke(tint.opacity(0.18), lineWidth: lineWidth)
            Circle()
                .trim(from: 0, to: max(0.001, min(progress, 1)))
                .stroke(tint, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .animation(Theme.Motion.smooth, value: progress)
    }
}

/// A line icon in its color on a faint wash of it.
struct IconTile: View {
    let systemImage: String
    let tint: Color
    var size: CGFloat = 22

    var body: some View {
        Image(systemName: systemImage)
            .font(.system(size: size * 0.55, weight: .medium))
            .foregroundStyle(tint)
            .frame(width: size, height: size)
            .background(tint.opacity(0.13), in: RoundedRectangle(cornerRadius: size * 0.3, style: .continuous))
    }
}

/// The lesson status glyph used in the sidebar and on the home screen.
struct StatusGlyph: View {
    let status: String

    var body: some View {
        Group {
            switch status {
            case "passed":
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.white, Theme.pass)
            case "in_progress":
                Image(systemName: "circle.lefthalf.filled")
                    .foregroundStyle(Theme.accent)
            default:
                Image(systemName: "circle")
                    .foregroundStyle(.tertiary)
            }
        }
        .contentTransition(.symbolEffect(.replace))
        .animation(Theme.Motion.bouncy, value: status)
    }
}
