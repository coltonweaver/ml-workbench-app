import AppKit
import SwiftUI
import WorkbenchCore

/// Renders the small Markdown subset lessons use: paragraphs with inline
/// formatting, headings, bullet and numbered lists, fenced code and pipe tables.
/// SwiftUI's own Markdown support is inline-only.
struct MarkdownView: View {
    let source: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(MarkdownBlock.parse(source).enumerated()), id: \.offset) { _, block in
                view(for: block)
            }
        }
        .font(Theme.body)
        .lineSpacing(Theme.bodyLineSpacing)
        .frame(maxWidth: .infinity, alignment: .leading)
        .textSelection(.enabled)
    }

    @ViewBuilder
    private func view(for block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let text):
            inline(text)
                .font(Theme.display(15, weight: .semibold))
                .padding(.top, 4)
        case .paragraph(let text):
            inline(text)
                .fixedSize(horizontal: false, vertical: true)
        case .list(let items, ordered: let ordered):
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(items.enumerated()), id: \.offset) { i, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        if ordered {
                            Text("\(i + 1).")
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                                .frame(minWidth: 16, alignment: .trailing)
                        } else {
                            Circle()
                                .fill(Color.secondary.opacity(0.7))
                                .frame(width: 5, height: 5)
                                .alignmentGuide(.firstTextBaseline) { $0[.bottom] + 4 }
                                .frame(width: 16)
                        }
                        inline(item)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        case .code(let text):
            CodeBlock(code: text)
        case .table(let rows):
            TableBlock(rows: rows, inline: inline)
        }
    }

    private func inline(_ text: String) -> Text {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        if let attributed = try? AttributedString(markdown: text, options: options) {
            return Text(attributed)
        }
        return Text(text)
    }
}

/// Fenced code, highlighted like the editor and scrolling sideways rather
/// than wrapping.
private struct CodeBlock: View {
    let code: String
    @State private var copied = false
    @State private var hovering = false

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Text(EditorTheme.attributed(code))
                .lineSpacing(2)
                .fixedSize(horizontal: true, vertical: true)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: EditorTheme.background), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        }
        .overlay(alignment: .topTrailing) {
            if hovering || copied {
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                    withAnimation(Theme.Motion.snappy) { copied = true }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.4) {
                        withAnimation(Theme.Motion.snappy) { copied = false }
                    }
                } label: {
                    Image(systemName: copied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 11, weight: .medium))
                        .contentTransition(.symbolEffect(.replace))
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.borderless)
                .glassSurface(Circle(), interactive: true)
                .padding(6)
                .transition(.opacity)
                .help("Copy")
            }
        }
        .onHover { inside in withAnimation(Theme.Motion.snappy) { hovering = inside } }
    }
}

private struct TableBlock: View {
    let rows: [[String]]
    let inline: (String) -> Text

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
            ForEach(Array(rows.enumerated()), id: \.offset) { i, row in
                GridRow {
                    ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                        inline(cell)
                            .font(i == 0 ? .system(size: 12, weight: .semibold) : .system(size: 13))
                            .foregroundStyle(i == 0 ? Color.secondary : Color.primary)
                            .monospacedDigit()
                            .padding(.horizontal, 12)
                            .padding(.vertical, 7)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(i == 0 ? Color.primary.opacity(0.05)
                                        : (i % 2 == 0 ? Color.primary.opacity(0.025) : Color.clear))
                    }
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        }
    }
}
