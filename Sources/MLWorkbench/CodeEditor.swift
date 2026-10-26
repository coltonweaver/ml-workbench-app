import AppKit
import SwiftUI
import WorkbenchCore

/// The code editor: an NSTextView (SwiftUI's TextEditor follows the system's
/// smart-quote and dash settings, which silently corrupts Python source) with
/// syntax highlighting, line numbers, Python-aware indentation and
/// completions from the sidecar.
struct CodeEditor: NSViewRepresentable {
    @Binding var text: String
    /// Asks for completions at a 1-based line and 0-based column (in Unicode
    /// scalars) of the given text.
    var complete: ((String, Int, Int) async -> AppModel.CompletionResult?)?

    func makeNSView(context: Context) -> NSScrollView {
        // TextKit 1, for the line-number ruler and current-line highlight.
        let storage = NSTextStorage()
        let layout = NSLayoutManager()
        storage.addLayoutManager(layout)
        let container = NSTextContainer(size: NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude))
        container.widthTracksTextView = true
        layout.addTextContainer(container)

        let view = CodeTextView(frame: .zero, textContainer: container)
        view.storage = storage
        storage.delegate = context.coordinator
        view.delegate = context.coordinator
        view.isRichText = false
        view.importsGraphics = false
        view.allowsUndo = true
        view.usesFindBar = true
        view.isIncrementalSearchingEnabled = true
        view.isAutomaticQuoteSubstitutionEnabled = false
        view.isAutomaticDashSubstitutionEnabled = false
        view.isAutomaticTextReplacementEnabled = false
        view.isAutomaticSpellingCorrectionEnabled = false
        view.isContinuousSpellCheckingEnabled = false
        view.isGrammarCheckingEnabled = false
        view.smartInsertDeleteEnabled = false
        view.isVerticallyResizable = true
        view.isHorizontallyResizable = false
        view.autoresizingMask = [.width]
        view.minSize = .zero
        view.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        view.textContainerInset = NSSize(width: 8, height: 12)
        view.font = EditorTheme.font
        view.textColor = EditorTheme.text
        view.backgroundColor = EditorTheme.background
        view.insertionPointColor = .controlAccentColor
        view.selectedTextAttributes = [.backgroundColor: EditorTheme.selection]
        view.defaultParagraphStyle = CodeTextView.paragraphStyle
        view.typingAttributes = CodeTextView.baseAttributes
        view.string = text

        let scroll = NSScrollView()
        scroll.documentView = view
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.drawsBackground = true
        scroll.backgroundColor = EditorTheme.background
        scroll.scrollerStyle = .overlay

        let ruler = LineNumberRuler(textView: view)
        scroll.verticalRulerView = ruler
        scroll.hasVerticalRuler = true
        scroll.rulersVisible = true

        context.coordinator.attach(view)
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.text = $text
        context.coordinator.complete = complete
        guard let view = scroll.documentView as? CodeTextView, view.string != text else { return }
        // Only replace the text when it changed from outside (file switch or
        // reload), so typing doesn't reset the cursor.
        context.coordinator.completion.dismiss()
        view.string = text
        view.setSelectedRange(NSRange(location: 0, length: 0))
        view.scrollToBeginningOfDocument(nil)
        scroll.verticalRulerView?.needsDisplay = true
    }

    static func dismantleNSView(_ scroll: NSScrollView, coordinator: Coordinator) {
        coordinator.completion.dismiss()
    }

    func makeCoordinator() -> Coordinator { Coordinator(text: $text, complete: complete) }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate, NSTextStorageDelegate {
        var text: Binding<String>
        var complete: ((String, Int, Int) async -> AppModel.CompletionResult?)?
        let completion = CompletionController()
        private weak var view: CodeTextView?

        init(text: Binding<String>, complete: ((String, Int, Int) async -> AppModel.CompletionResult?)?) {
            self.text = text
            self.complete = complete
        }

        func attach(_ view: CodeTextView) {
            self.view = view
            completion.textView = view
            completion.fetch = { [weak self] code, line, column in
                await self?.complete?(code, line, column)
            }
            view.onCompleteRequested = { [weak self] in self?.completion.trigger(manual: true) }
        }

        // MARK: Highlighting

        nonisolated func textStorage(
            _ textStorage: NSTextStorage, didProcessEditing editedMask: NSTextStorageEditActions,
            range editedRange: NSRange, changeInLength delta: Int
        ) {
            guard editedMask.contains(.editedCharacters) else { return }
            MainActor.assumeIsolated { Self.highlight(textStorage) }
        }

        static func highlight(_ storage: NSTextStorage) {
            let full = NSRange(location: 0, length: storage.length)
            storage.setAttributes(CodeTextView.baseAttributes, range: full)
            for token in PythonTokenizer.tokenize(storage.string) where NSMaxRange(token.range) <= full.length {
                storage.addAttribute(.foregroundColor, value: EditorTheme.color(for: token.kind), range: token.range)
                if let font = EditorTheme.font(for: token.kind) {
                    storage.addAttribute(.font, value: font, range: token.range)
                }
            }
        }

        // MARK: Editing

        func textDidChange(_ notification: Notification) {
            guard let view = notification.object as? CodeTextView else { return }
            text.wrappedValue = view.string
            view.enclosingScrollView?.verticalRulerView?.needsDisplay = true
            completion.textChanged(typed: view.lastTyped)
            view.lastTyped = nil
        }

        func textViewDidChangeSelection(_ notification: Notification) {
            guard let view = notification.object as? CodeTextView else { return }
            view.needsDisplay = true
            view.enclosingScrollView?.verticalRulerView?.needsDisplay = true
            completion.selectionChanged()
        }

        func textView(_ textView: NSTextView, doCommandBy selector: Selector) -> Bool {
            completion.handle(selector)
        }
    }
}

// MARK: - Text view

final class CodeTextView: NSTextView {
    /// TextKit 1 views don't retain their text storage.
    var storage: NSTextStorage?
    /// The text typed by the last key press, for completion triggers.
    var lastTyped: String?
    var onCompleteRequested: (() -> Void)?

    static let indent = "    "

    static let paragraphStyle: NSParagraphStyle = {
        let style = NSMutableParagraphStyle()
        style.lineHeightMultiple = 1.18
        let tabWidth = (" " as NSString).size(withAttributes: [.font: EditorTheme.font]).width * 4
        style.defaultTabInterval = tabWidth
        style.tabStops = []
        return style
    }()

    static let baseAttributes: [NSAttributedString.Key: Any] = [
        .font: EditorTheme.font,
        .foregroundColor: EditorTheme.text,
        .paragraphStyle: paragraphStyle,
    ]

    private var nsString: NSString { string as NSString }

    // Current line highlight
    override func drawBackground(in rect: NSRect) {
        super.drawBackground(in: rect)
        guard let layoutManager, let textContainer, selectedRange().length == 0 else { return }
        let caret = selectedRange().location
        var lineRect: NSRect
        if caret >= nsString.length, layoutManager.extraLineFragmentTextContainer != nil {
            lineRect = layoutManager.extraLineFragmentRect
        } else {
            let lineRange = nsString.lineRange(for: NSRange(location: min(caret, max(0, nsString.length - 1)), length: 0))
            let glyphs = layoutManager.glyphRange(forCharacterRange: lineRange, actualCharacterRange: nil)
            lineRect = layoutManager.boundingRect(forGlyphRange: glyphs, in: textContainer)
        }
        lineRect.origin.x = 0
        lineRect.size.width = bounds.width
        lineRect.origin.y += textContainerOrigin.y
        EditorTheme.currentLine.setFill()
        lineRect.fill()
    }

    // MARK: Indentation

    /// The leading whitespace of the line containing `location`.
    private func indentation(at location: Int) -> String {
        let line = nsString.substring(with: nsString.lineRange(for: NSRange(location: location, length: 0)))
        return String(line.prefix { $0 == " " || $0 == "\t" })
    }

    override func insertNewline(_ sender: Any?) {
        let caret = selectedRange().location
        let lineRange = nsString.lineRange(for: NSRange(location: caret, length: 0))
        let beforeCaret = nsString.substring(with: NSRange(location: lineRange.location, length: caret - lineRange.location))
        var indent = indentation(at: caret)
        let trimmed = beforeCaret.trimmingCharacters(in: .whitespaces)
        if trimmed.hasSuffix(":") || trimmed.hasSuffix("(") || trimmed.hasSuffix("[") || trimmed.hasSuffix("{") {
            indent += Self.indent
        } else if ["return", "pass", "break", "continue", "raise"].contains(where: { trimmed == $0 || trimmed.hasPrefix($0 + " ") }),
                  indent.count >= Self.indent.count {
            indent.removeLast(Self.indent.count)
        }
        if beforeCaret.trimmingCharacters(in: .whitespaces).isEmpty {
            // On a blank line, keep its own indentation rather than adding to it.
            indent = String(beforeCaret.prefix { $0 == " " || $0 == "\t" })
        }
        insertText("\n" + indent, replacementRange: selectedRange())
    }

    override func insertTab(_ sender: Any?) {
        if selectedRange().length > 0, selectionSpansLines {
            shiftSelectedLines(by: 1)
        } else {
            let caret = selectedRange().location
            let column = caret - nsString.lineRange(for: NSRange(location: caret, length: 0)).location
            insertText(String(repeating: " ", count: 4 - column % 4), replacementRange: selectedRange())
        }
    }

    override func insertBacktab(_ sender: Any?) {
        shiftSelectedLines(by: -1)
    }

    /// Backspace in leading whitespace removes back to the previous indent stop.
    override func deleteBackward(_ sender: Any?) {
        lastTyped = nil
        let range = selectedRange()
        if range.length == 0, range.location > 0 {
            let lineStart = nsString.lineRange(for: NSRange(location: range.location, length: 0)).location
            let before = nsString.substring(with: NSRange(location: lineStart, length: range.location - lineStart))
            if !before.isEmpty, before.allSatisfy({ $0 == " " }) {
                let remove = before.count % 4 == 0 ? 4 : before.count % 4
                let target = NSRange(location: range.location - remove, length: remove)
                if shouldChangeText(in: target, replacementString: "") {
                    replaceCharacters(in: target, with: "")
                    didChangeText()
                }
                return
            }
            // Delete an empty pair together: ( ) [ ] { } '' ""
            if range.location < nsString.length {
                let pair = nsString.substring(with: NSRange(location: range.location - 1, length: 2))
                if ["()", "[]", "{}", "''", "\"\""].contains(pair) {
                    let target = NSRange(location: range.location - 1, length: 2)
                    if shouldChangeText(in: target, replacementString: "") {
                        replaceCharacters(in: target, with: "")
                        didChangeText()
                    }
                    return
                }
            }
        }
        super.deleteBackward(sender)
    }

    private var selectionSpansLines: Bool {
        nsString.substring(with: selectedRange()).contains("\n")
    }

    private func shiftSelectedLines(by direction: Int) {
        let lines = nsString.lineRange(for: selectedRange())
        let original = nsString.substring(with: lines)
        var parts = original.components(separatedBy: "\n")
        let trailingNewline = parts.last == ""
        if trailingNewline { parts.removeLast() }
        let shifted = parts.map { line -> String in
            if direction > 0 { return line.isEmpty ? line : Self.indent + line }
            let spaces = line.prefix(4).prefix { $0 == " " }.count
            return String(line.dropFirst(spaces))
        }
        let replacement = shifted.joined(separator: "\n") + (trailingNewline ? "\n" : "")
        guard shouldChangeText(in: lines, replacementString: replacement) else { return }
        replaceCharacters(in: lines, with: replacement)
        didChangeText()
        setSelectedRange(NSRange(location: lines.location, length: (replacement as NSString).length))
    }

    /// ⌘/ toggles `# ` on the selected lines.
    func toggleComment() {
        let lines = nsString.lineRange(for: selectedRange())
        var parts = nsString.substring(with: lines).components(separatedBy: "\n")
        let trailingNewline = parts.last == ""
        if trailingNewline { parts.removeLast() }
        let code = parts.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        let uncomment = !code.isEmpty && code.allSatisfy { $0.trimmingCharacters(in: .whitespaces).hasPrefix("#") }
        let minIndent = code.map { $0.prefix { $0 == " " }.count }.min() ?? 0
        let changed = parts.map { line -> String in
            if line.trimmingCharacters(in: .whitespaces).isEmpty { return line }
            if uncomment {
                guard let hash = line.firstIndex(of: "#") else { return line }
                var rest = line[line.index(after: hash)...]
                if rest.hasPrefix(" ") { rest = rest.dropFirst() }
                return String(line[..<hash]) + rest
            }
            let index = line.index(line.startIndex, offsetBy: minIndent)
            return String(line[..<index]) + "# " + line[index...]
        }
        let replacement = changed.joined(separator: "\n") + (trailingNewline ? "\n" : "")
        guard shouldChangeText(in: lines, replacementString: replacement) else { return }
        replaceCharacters(in: lines, with: replacement)
        didChangeText()
        setSelectedRange(NSRange(location: lines.location, length: (replacement as NSString).length))
    }

    // MARK: Brackets

    private static let pairs: [String: String] = ["(": ")", "[": "]", "{": "}"]
    private static let closers: Set<String> = [")", "]", "}", "\"", "'"]

    override func insertText(_ string: Any, replacementRange: NSRange) {
        let typed = (string as? String) ?? (string as? NSAttributedString)?.string
        lastTyped = typed
        guard let typed, replacementRange.location == NSNotFound, hasMarkedText() == false else {
            super.insertText(string, replacementRange: replacementRange)
            return
        }
        let range = selectedRange()
        let next = range.location < nsString.length && range.length == 0
            ? nsString.substring(with: NSRange(location: range.location, length: 1)) : ""
        // Typing a closer that's already there steps over it.
        if Self.closers.contains(typed), next == typed, range.length == 0 {
            lastTyped = nil
            setSelectedRange(NSRange(location: range.location + 1, length: 0))
            return
        }
        // Opening brackets get their pair when nothing word-like follows.
        if let close = Self.pairs[typed], range.length == 0,
           next.isEmpty || next == "\n" || next == " " || Self.closers.contains(next) || next == "," || next == ":" {
            super.insertText(typed + close, replacementRange: replacementRange)
            setSelectedRange(NSRange(location: range.location + 1, length: 0))
            return
        }
        super.insertText(string, replacementRange: replacementRange)
    }

    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        if flags == .command, event.charactersIgnoringModifiers == "/" {
            toggleComment()
            return true
        }
        return super.performKeyEquivalent(with: event)
    }

    override func keyDown(with event: NSEvent) {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        // ⌃Space asks for completions, as in most editors.
        if flags == .control, event.charactersIgnoringModifiers == " " {
            onCompleteRequested?()
            return
        }
        super.keyDown(with: event)
    }

    /// Esc and ⌥Esc (NSTextView's own completion keys) show our completions.
    override func complete(_ sender: Any?) {
        onCompleteRequested?()
    }

    /// Screen rect of the caret, for placing the completion popup.
    func caretScreenRect() -> NSRect? {
        let range = NSRange(location: selectedRange().location, length: 0)
        let rect = firstRect(forCharacterRange: range, actualRange: nil)
        return rect == .zero ? nil : rect
    }
}

// MARK: - Line numbers

final class LineNumberRuler: NSRulerView {
    private weak var textView: NSTextView?

    init(textView: NSTextView) {
        self.textView = textView
        super.init(scrollView: textView.enclosingScrollView, orientation: .verticalRuler)
        clientView = textView
        ruleThickness = 44
        let clip = textView.enclosingScrollView?.contentView
        clip?.postsBoundsChangedNotifications = true
        NotificationCenter.default.addObserver(
            self, selector: #selector(boundsChanged), name: NSView.boundsDidChangeNotification, object: clip)
    }

    required init(coder: NSCoder) { fatalError("not used") }

    override var isFlipped: Bool { true }

    @objc private func boundsChanged(_ note: Notification) {
        needsDisplay = true
    }

    override func drawHashMarksAndLabels(in rect: NSRect) {
        EditorTheme.gutter.setFill()
        bounds.fill()
        guard let textView, let layoutManager = textView.layoutManager, let container = textView.textContainer,
              let scrollView else { return }
        let text = textView.string as NSString
        let visible = scrollView.contentView.bounds
        let glyphs = layoutManager.glyphRange(forBoundingRect: visible, in: container)
        let chars = layoutManager.characterRange(forGlyphRange: glyphs, actualGlyphRange: nil)

        // Keep the gutter wide enough for the biggest line number.
        let lineCount = textView.string.utf16.reduce(1) { $1 == 10 ? $0 + 1 : $0 }
        let digits = max(2, String(lineCount).count)
        let font = NSFont.monospacedDigitSystemFont(ofSize: EditorTheme.fontSize - 2, weight: .regular)
        let digitWidth = ("8" as NSString).size(withAttributes: [.font: font]).width
        let wanted = ceil(CGFloat(digits) * digitWidth + 24)
        if abs(wanted - ruleThickness) > 0.5 {
            ruleThickness = wanted
            // The scroll view re-tiles; draw again at the new width.
            DispatchQueue.main.async { [weak self] in self?.needsDisplay = true }
            return
        }

        let caretLine = lineNumber(at: textView.selectedRange().location, in: text)
        var line = lineNumber(at: chars.location, in: text)
        var index = text.lineRange(for: NSRange(location: chars.location, length: 0)).location
        let inset = textView.textContainerOrigin.y

        func draw(_ number: Int, fragment: NSRect) {
            let current = number == caretLine
            let attrs: [NSAttributedString.Key: Any] = [
                .font: current ? NSFont.monospacedDigitSystemFont(ofSize: EditorTheme.fontSize - 2, weight: .semibold) : font,
                .foregroundColor: current ? EditorTheme.gutterCurrent : EditorTheme.gutterText,
            ]
            let label = "\(number)" as NSString
            let size = label.size(withAttributes: attrs)
            let y = convert(NSPoint(x: 0, y: fragment.minY + inset), from: textView).y
            // Align with the code's baseline: center within the line's height.
            let lineHeight = layoutManager.defaultLineHeight(for: EditorTheme.font)
            let point = NSPoint(x: ruleThickness - size.width - 12,
                                y: y + (fragment.height - lineHeight) + (lineHeight - size.height) / 2)
            label.draw(at: point, withAttributes: attrs)
        }

        while text.length > 0, index < NSMaxRange(chars) {
            let lineRange = text.lineRange(for: NSRange(location: index, length: 0))
            let glyph = layoutManager.glyphIndexForCharacter(at: lineRange.location)
            let fragment = layoutManager.lineFragmentRect(forGlyphAt: glyph, effectiveRange: nil)
            draw(line, fragment: fragment)
            line += 1
            index = NSMaxRange(lineRange)
            if index >= text.length { break }
        }
        if layoutManager.extraLineFragmentTextContainer != nil {
            draw(lineCount, fragment: layoutManager.extraLineFragmentRect)
        }
    }

    private func lineNumber(at location: Int, in text: NSString) -> Int {
        var count = 1
        var i = 0
        let end = min(location, text.length)
        while i < end {
            let range = text.range(of: "\n", options: [], range: NSRange(location: i, length: end - i))
            if range.location == NSNotFound { break }
            count += 1
            i = range.location + 1
        }
        return count
    }
}
