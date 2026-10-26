import AppKit
import SwiftUI

/// Spacing, radii, type and color in one place, so every screen shares them.
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
        static let card: CGFloat = 16
        static let large: CGFloat = 22
    }

    enum Motion {
        static let snappy = Animation.snappy(duration: 0.28)
        static let smooth = Animation.smooth(duration: 0.4)
        static let bouncy = Animation.bouncy(duration: 0.5, extraBounce: 0.1)
    }

    /// Lesson stages get their own hue, used for icons, tints and the stepper.
    enum Stage {
        static let problem = Color.blue
        static let predict = Color.purple
        static let build = Color.orange
        static let check = Color.green
        static let explain = Color.yellow
        static let stretch = Color.teal
    }

    static let pass = Color.green
    static let warn = Color.orange
    static let fail = Color.red

    /// Reading text in the lesson pane.
    static let body = Font.system(size: 14)
    static let bodyLineSpacing: CGFloat = 3.5
    static func display(_ size: CGFloat, weight: Font.Weight = .bold) -> Font {
        .system(size: size, weight: weight, design: .rounded)
    }
    static let eyebrow = Font.system(size: 11, weight: .semibold).width(.expanded)
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

/// A quiet card for content: a soft fill that reads in light and dark, with
/// a hairline edge. Glass is kept for things that float (controls, popovers),
/// as the platform intends; reading surfaces stay calm.
struct CardSurface: ViewModifier {
    var tint: Color?
    var radius: CGFloat = Theme.Radius.card

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
        content
            .background {
                ZStack {
                    shape.fill(Color(light: NSColor(white: 1, alpha: 0.75), dark: NSColor(white: 1, alpha: 0.045)))
                    if let tint {
                        shape.fill(LinearGradient(colors: [tint.opacity(0.13), tint.opacity(0.04)],
                                                  startPoint: .topLeading, endPoint: .bottomTrailing))
                    }
                }
            }
            .overlay {
                shape.stroke(Color(light: NSColor(white: 0, alpha: 0.07), dark: NSColor(white: 1, alpha: 0.08)),
                             lineWidth: 0.5)
            }
            .shadow(color: .black.opacity(0.06), radius: 8, y: 2)
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

    /// The prominent action: glass on macOS 26, bordered prominent before.
    @ViewBuilder
    func prominentButton() -> some View {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            buttonStyle(.glassProminent)
        } else {
            buttonStyle(.borderedProminent)
        }
        #else
        buttonStyle(.borderedProminent)
        #endif
    }

    /// A secondary action: glass on macOS 26, bordered before.
    @ViewBuilder
    func secondaryButton() -> some View {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            buttonStyle(.glass)
        } else {
            buttonStyle(.bordered)
        }
        #else
        buttonStyle(.bordered)
        #endif
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

/// A small uppercase label above a title.
struct Eyebrow: View {
    let text: String
    var color: Color = .secondary

    init(_ text: String, color: Color = .secondary) {
        self.text = text
        self.color = color
    }

    var body: some View {
        Text(text.uppercased())
            .font(Theme.eyebrow)
            .tracking(0.6)
            .foregroundStyle(color)
    }
}

/// A circular progress ring.
struct ProgressRing: View {
    var progress: Double
    var lineWidth: CGFloat = 4
    var tint: Color = .accentColor

    var body: some View {
        ZStack {
            Circle().stroke(tint.opacity(0.18), lineWidth: lineWidth)
            Circle()
                .trim(from: 0, to: max(0.001, min(progress, 1)))
                .stroke(tint.gradient, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .animation(Theme.Motion.smooth, value: progress)
    }
}

/// An icon in a tinted rounded square, like System Settings rows.
struct IconTile: View {
    let systemImage: String
    let tint: Color
    var size: CGFloat = 22

    var body: some View {
        Image(systemName: systemImage)
            .font(.system(size: size * 0.52, weight: .semibold))
            .foregroundStyle(.white)
            .frame(width: size, height: size)
            .background(tint.gradient, in: RoundedRectangle(cornerRadius: size * 0.28, style: .continuous))
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
                    .foregroundStyle(Color.accentColor)
            default:
                Image(systemName: "circle")
                    .foregroundStyle(.tertiary)
            }
        }
        .contentTransition(.symbolEffect(.replace))
        .animation(Theme.Motion.bouncy, value: status)
    }
}
