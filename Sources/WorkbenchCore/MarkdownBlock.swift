import Foundation

/// The block structure of the Markdown subset lessons use. Parsed here, rendered
/// by the app's MarkdownView.
public enum MarkdownBlock: Equatable, Sendable {
    case heading(String)
    case paragraph(String)
    case list([String], ordered: Bool)
    case code(String)
    case table([[String]])

    public static func parse(_ source: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        var listItems: [String] = []
        var listOrdered = false
        var table: [[String]] = []
        var code: [String]?

        func flush() {
            if !paragraph.isEmpty {
                blocks.append(.paragraph(paragraph.joined(separator: " ")))
                paragraph = []
            }
            if !listItems.isEmpty {
                blocks.append(.list(listItems, ordered: listOrdered))
                listItems = []
            }
            if !table.isEmpty {
                blocks.append(.table(table))
                table = []
            }
        }

        for raw in source.components(separatedBy: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if var lines = code {
                if line.hasPrefix("```") {
                    blocks.append(.code(lines.joined(separator: "\n")))
                    code = nil
                } else {
                    lines.append(raw)
                    code = lines
                }
                continue
            }
            if line.hasPrefix("```") {
                flush()
                code = []
            } else if line.isEmpty {
                flush()
            } else if line.hasPrefix("#") {
                flush()
                blocks.append(.heading(String(line.drop(while: { $0 == "#" })).trimmingCharacters(in: .whitespaces)))
            } else if line.hasPrefix("|") {
                if !paragraph.isEmpty || !listItems.isEmpty { flush() }
                let cells = line.trimmingCharacters(in: CharacterSet(charactersIn: "|"))
                    .components(separatedBy: "|")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                // Skip the |---|---| separator row.
                if !cells.allSatisfy({ $0.allSatisfy { "-: ".contains($0) } }) {
                    table.append(cells)
                }
            } else if let item = listItem(line) {
                if !paragraph.isEmpty || !table.isEmpty { flush() }
                if !listItems.isEmpty && listOrdered != item.ordered { flush() }
                listOrdered = item.ordered
                listItems.append(item.text)
            } else if !listItems.isEmpty && raw.hasPrefix(" ") {
                // Continuation of the previous list item.
                listItems[listItems.count - 1] += " " + line
            } else {
                if !listItems.isEmpty || !table.isEmpty { flush() }
                paragraph.append(line)
            }
        }
        if let lines = code { blocks.append(.code(lines.joined(separator: "\n"))) }
        flush()
        return blocks
    }

    private static func listItem(_ line: String) -> (text: String, ordered: Bool)? {
        if line.hasPrefix("- ") || line.hasPrefix("* ") {
            return (String(line.dropFirst(2)), false)
        }
        let digits = line.prefix(while: \.isNumber)
        if !digits.isEmpty, line.dropFirst(digits.count).hasPrefix(". ") {
            return (String(line.dropFirst(digits.count + 2)), true)
        }
        return nil
    }
}
