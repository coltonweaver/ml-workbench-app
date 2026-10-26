import AppKit
import SwiftUI
import WorkbenchCore

/// Colors and fonts for code, shared by the editor and code blocks in
/// lessons. Close to Xcode's default light and dark themes.
enum EditorTheme {
    static let fontSize: CGFloat = 13
    static let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: .regular)
    static let boldFont = NSFont.monospacedSystemFont(ofSize: fontSize, weight: .semibold)
    static let italicFont: NSFont = {
        let descriptor = font.fontDescriptor.withSymbolicTraits(.italic)
        return NSFont(descriptor: descriptor, size: fontSize) ?? font
    }()

    static let text = NSColor.dynamic(light: 0x1F1F24, dark: 0xDCDCDF)
    static let background = NSColor.dynamic(light: 0xFFFFFF, dark: 0x1C1D21)
    static let gutter = NSColor.dynamic(light: 0xF7F7F9, dark: 0x1C1D21)
    static let gutterText = NSColor.dynamic(light: 0xA9AAB0, dark: 0x5A5D66)
    static let gutterCurrent = NSColor.dynamic(light: 0x4A4B52, dark: 0xC9CAD0)
    static let currentLine = NSColor.dynamic(light: 0xF1F5FD, dark: 0x25272D)
    static let selection = NSColor.dynamic(light: 0xC9DDFB, dark: 0x3A4A66)

    static func color(for kind: PythonTokenizer.Kind) -> NSColor {
        switch kind {
        case .keyword, .constant, .selfParam: .dynamic(light: 0xAD3DA4, dark: 0xFC5FA3)
        case .builtin: .dynamic(light: 0x6C36A9, dark: 0xA167E6)
        case .number: .dynamic(light: 0x1C00CF, dark: 0xD0BF69)
        case .string: .dynamic(light: 0xC41A16, dark: 0xFC6A5D)
        case .comment: .dynamic(light: 0x6C7883, dark: 0x7F8C98)
        case .decorator: .dynamic(light: 0x815F03, dark: 0xFD8F3F)
        case .definition: .dynamic(light: 0x0F68A0, dark: 0x41A1C0)
        case .type: .dynamic(light: 0x0B4F79, dark: 0x5DD8FF)
        case .call: .dynamic(light: 0x326D74, dark: 0x67B7A4)
        }
    }

    static func font(for kind: PythonTokenizer.Kind) -> NSFont? {
        switch kind {
        case .keyword, .constant: boldFont
        case .comment, .selfParam: italicFont
        default: nil
        }
    }

    /// Highlighted code for SwiftUI `Text`, e.g. code blocks in lessons.
    static func attributed(_ code: String, size: CGFloat = 12.5) -> AttributedString {
        // Keyed by attribute type: `.font` alone is ambiguous between the
        // SwiftUI and AppKit scopes.
        typealias UI = AttributeScopes.SwiftUIAttributes
        var out = AttributedString(code)
        out[UI.FontAttribute.self] = .system(size: size, design: .monospaced)
        out[UI.ForegroundColorAttribute.self] = Color(nsColor: text)
        for token in PythonTokenizer.tokenize(code) {
            guard let range = Range(token.range, in: out) else { continue }
            out[range][UI.ForegroundColorAttribute.self] = Color(nsColor: color(for: token.kind))
            if token.kind == .keyword || token.kind == .constant {
                out[range][UI.FontAttribute.self] = .system(size: size, weight: .semibold, design: .monospaced)
            }
        }
        return out
    }
}
