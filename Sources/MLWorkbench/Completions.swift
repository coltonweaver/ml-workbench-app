import AppKit
import SwiftUI
import WorkbenchCore

/// Drives the editor's completion popup and parameter hint: decides when to
/// ask, filters as you type, and handles the keys while the popup is up.
/// Names from the file, keywords and builtins show at once; the sidecar's
/// Jedi completions replace them when they arrive.
@MainActor
final class CompletionController {
    weak var textView: CodeTextView?
    var fetch: ((String, Int, Int) async -> AppModel.CompletionResult?)?

    private enum Mode { case completions, signature }

    private let listHost: NSHostingView<CompletionListView>
    private let hintHost: NSHostingView<SignatureHintView>
    private let listPanel: FloatingPanel
    private let hintPanel: FloatingPanel

    /// Where the word being completed starts.
    private var anchor: Int?
    private var isMember = false
    private var remote: [Completion] = []
    private var remoteAnchor: Int?
    private var items: [Completion] = []
    private var selected = 0
    private var hintLine: Int?
    private var task: Task<Void, Never>?
    /// Set while we edit the text ourselves, so it doesn't re-trigger.
    private var applying = false

    static let listWidth: CGFloat = 460
    static let rowHeight: CGFloat = 24
    static let visibleRows = 9
    static let detailHeight: CGFloat = 76

    init() {
        listHost = NSHostingView(rootView: CompletionListView(items: [], selected: 0, prefix: "", onPick: { _ in }))
        hintHost = NSHostingView(rootView: SignatureHintView(signature: nil))
        listPanel = FloatingPanel(content: listHost, cornerRadius: 12)
        hintPanel = FloatingPanel(content: hintHost, cornerRadius: 10)
    }

    var isVisible: Bool { listPanel.isVisible }

    // MARK: Triggers

    func textChanged(typed: String?) {
        guard !applying, let view = textView else { return }
        let text = view.string as NSString
        let caret = view.selectedRange().location
        let word = wordStart(before: caret, in: text)
        let member = word > 0 && text.character(at: word - 1) == 46  // "."

        guard let typed else {
            // A deletion: follow it if a popup is up.
            if isVisible || anchor != nil {
                if caret <= (anchor ?? 0) && !(member && caret == word) {
                    dismiss()
                } else {
                    anchor = word
                    isMember = member
                    refilter()
                    request(.completions)
                }
            }
            return
        }

        if typed.contains("\n") || typed == ")" {
            dismiss()
            hideHint()
            return
        }
        if typed == "(" || typed == "," {
            dismiss()
            request(.signature)
            return
        }
        if typed == "." {
            // Not after a digit (1.5) or inside a string or comment.
            let afterDigit = caret >= 2 && (48...57).contains(text.character(at: caret - 2))
            if afterDigit || insideStringOrComment(at: caret, in: view.string) {
                dismiss()
                return
            }
            anchor = caret
            isMember = true
            remote = []
            remoteAnchor = nil
            items = []
            hideList()
            request(.completions)
            return
        }
        if let last = typed.unicodeScalars.last, isIdentifier(UInt16(truncatingIfNeeded: last.value)) || last.value > 0x7F {
            let prefixLength = caret - word
            if member || prefixLength >= 2 || isVisible {
                if anchor != word {
                    remote = []
                    remoteAnchor = nil
                }
                anchor = word
                isMember = member
                // Inside a string or comment there's nothing to complete.
                if insideStringOrComment(at: caret, in: view.string) {
                    dismiss()
                    return
                }
                refilter()
                request(.completions)
            }
            return
        }
        dismiss()
    }

    /// ⌃Space, Esc or ⌥Esc.
    func trigger(manual: Bool) {
        guard let view = textView else { return }
        let text = view.string as NSString
        let caret = view.selectedRange().location
        let word = wordStart(before: caret, in: text)
        anchor = word
        isMember = word > 0 && text.character(at: word - 1) == 46
        if remoteAnchor != word { remote = [] }
        refilter(showEmpty: manual)
        request(.completions, delay: 0)
    }

    func selectionChanged() {
        guard !applying, let view = textView else { return }
        let caret = view.selectedRange().location
        if let anchor, isVisible {
            let text = view.string as NSString
            if view.selectedRange().length > 0 || caret < anchor || wordStart(before: caret, in: text) != anchor {
                dismiss()
            }
        }
        if let hintLine, line(at: caret, in: view.string as NSString) != hintLine {
            hideHint()
        }
    }

    // MARK: Keys

    func handle(_ selector: Selector) -> Bool {
        if isVisible {
            switch selector {
            case #selector(NSResponder.moveUp(_:)):
                move(by: -1)
                return true
            case #selector(NSResponder.moveDown(_:)):
                move(by: 1)
                return true
            case #selector(NSResponder.pageUp(_:)), #selector(NSResponder.scrollPageUp(_:)):
                move(by: -Self.visibleRows)
                return true
            case #selector(NSResponder.pageDown(_:)), #selector(NSResponder.scrollPageDown(_:)):
                move(by: Self.visibleRows)
                return true
            case #selector(NSResponder.insertNewline(_:)), #selector(NSResponder.insertTab(_:)):
                accept(selected)
                return true
            case #selector(NSResponder.cancelOperation(_:)):
                dismiss()
                return true
            case #selector(NSResponder.moveLeft(_:)), #selector(NSResponder.moveRight(_:)),
                 #selector(NSResponder.moveToBeginningOfLine(_:)), #selector(NSResponder.moveToEndOfLine(_:)):
                dismiss()
                return false
            default:
                return false
            }
        }
        if hintPanel.isVisible, selector == #selector(NSResponder.cancelOperation(_:)) {
            hideHint()
            return true
        }
        return false
    }

    private func move(by delta: Int) {
        guard !items.isEmpty else { return }
        selected = min(max(selected + delta, 0), items.count - 1)
        renderList()
    }

    // MARK: Requests

    private func request(_ mode: Mode, delay: Int = 90) {
        guard let view = textView, let fetch else { return }
        let code = view.string
        let caret = view.selectedRange().location
        let (lineNumber, column) = position(of: caret, in: code as NSString)
        let anchorAtRequest = anchor
        task?.cancel()
        task = Task { [weak self] in
            if delay > 0 { try? await Task.sleep(for: .milliseconds(delay)) }
            guard !Task.isCancelled else { return }
            let result = await fetch(code, lineNumber, column)
            guard !Task.isCancelled, let self, let result else { return }
            self.receive(result, mode: mode, anchor: anchorAtRequest, line: lineNumber)
        }
    }

    private func receive(_ result: AppModel.CompletionResult, mode: Mode, anchor requested: Int?, line: Int) {
        if let signature = result.signature {
            showHint(signature, line: line)
        } else if mode == .signature {
            hideHint()
        }
        guard mode == .completions, let anchor, requested == anchor else { return }
        remote = result.completions
        remoteAnchor = anchor
        refilter()
    }

    // MARK: Filtering

    private func refilter(showEmpty: Bool = false) {
        guard let view = textView, let anchor else { return dismiss() }
        let text = view.string as NSString
        let caret = view.selectedRange().location
        guard caret >= anchor, caret <= text.length else { return dismiss() }
        let prefix = text.substring(with: NSRange(location: anchor, length: caret - anchor))

        var candidates = remoteAnchor == anchor ? remote : []
        if !isMember {
            let seen = Set(candidates.map(\.name))
            candidates += localCompletions(in: view.string, excluding: prefix).filter { !seen.contains($0.name) }
        }
        let ranked = Self.rank(candidates, prefix: prefix)
        let previous = items.indices.contains(selected) ? items[selected].name : nil
        items = ranked
        selected = previous.flatMap { name in ranked.firstIndex { $0.name == name } } ?? 0
        if items.isEmpty || (items.count == 1 && items[0].name == prefix && !showEmpty) {
            hideList()
        } else {
            renderList()
        }
    }

    /// Prefix matches first (exact case, then any case), then the rest that
    /// contain the typed letters in order. The source's order is kept within
    /// each group.
    static func rank(_ candidates: [Completion], prefix: String) -> [Completion] {
        guard !prefix.isEmpty else { return candidates }
        let lower = prefix.lowercased()
        var exact: [Completion] = [], caseless: [Completion] = [], fuzzy: [Completion] = []
        for c in candidates {
            if c.name.hasPrefix(prefix) {
                exact.append(c)
            } else if c.name.lowercased().hasPrefix(lower) {
                caseless.append(c)
            } else if isSubsequence(lower, of: c.name.lowercased()) {
                fuzzy.append(c)
            }
        }
        return exact + caseless + fuzzy
    }

    private static func isSubsequence(_ needle: String, of haystack: String) -> Bool {
        var it = haystack.makeIterator()
        for ch in needle {
            var found = false
            while let h = it.next() {
                if h == ch {
                    found = true
                    break
                }
            }
            if !found { return false }
        }
        return true
    }

    private func localCompletions(in code: String, excluding typed: String) -> [Completion] {
        var names: [String] = []
        var seen = Set<String>()
        var current = ""
        func flush() {
            if current.count >= 3, current != typed, !seen.contains(current), !(current.first?.isNumber ?? true) {
                seen.insert(current)
                names.append(current)
            }
            current = ""
        }
        for ch in code {
            if ch.isLetter || ch.isNumber || ch == "_" { current.append(ch) } else { flush() }
        }
        flush()
        var out = names.filter { !PythonTokenizer.keywords.contains($0) }.map {
            Completion(name: $0, complete: "", type: "statement")
        }
        out += PythonTokenizer.keywords.sorted().map { Completion(name: $0, complete: "", type: "keyword") }
        out += PythonTokenizer.builtins.sorted().map {
            Completion(name: $0, complete: "", type: $0.first?.isUppercase == true ? "class" : "function")
        }
        return out
    }

    // MARK: Accepting

    private func accept(_ index: Int) {
        guard let view = textView, let anchor, items.indices.contains(index) else { return dismiss() }
        let item = items[index]
        let text = view.string as NSString
        var end = view.selectedRange().location
        while end < text.length, isIdentifier(text.character(at: end)) { end += 1 }
        let range = NSRange(location: anchor, length: end - anchor)
        let next = end < text.length ? text.character(at: end) : 0
        let callable = item.type == "function" || item.type == "class"
        let insertion = callable && next != 40 ? item.name + "()" : item.name

        dismiss()
        applying = true
        if view.shouldChangeText(in: range, replacementString: insertion) {
            view.replaceCharacters(in: range, with: insertion)
            view.didChangeText()
            let caret = anchor + (item.name as NSString).length + (insertion.hasSuffix("()") ? 1 : 0)
            view.setSelectedRange(NSRange(location: caret, length: 0))
        }
        applying = false
        if insertion.hasSuffix("()") { request(.signature, delay: 0) }
    }

    func dismiss() {
        task?.cancel()
        anchor = nil
        items = []
        selected = 0
        hideList()
    }

    // MARK: Panels

    private func renderList() {
        guard let view = textView, let window = view.window, let caret = view.caretScreenRect() else { return }
        let prefix: String = {
            guard let anchor else { return "" }
            let text = view.string as NSString
            let caret = view.selectedRange().location
            return caret >= anchor ? text.substring(with: NSRange(location: anchor, length: caret - anchor)) : ""
        }()
        listHost.rootView = CompletionListView(items: items, selected: selected, prefix: prefix) { [weak self] i in
            self?.accept(i)
        }
        let rows = min(items.count, Self.visibleRows)
        let hasDetail = items.indices.contains(selected) && (items[selected].signature != nil || items[selected].doc != nil)
        let height = CGFloat(rows) * Self.rowHeight + 12 + (hasDetail ? Self.detailHeight : 0)
        let size = NSSize(width: Self.listWidth, height: height)
        // Line the names up under the word being typed.
        var origin = NSPoint(x: caret.minX - 34 - prefixWidth(prefix), y: caret.minY - height - 6)
        if let screen = window.screen?.visibleFrame {
            if origin.y < screen.minY { origin.y = caret.maxY + 6 }
            origin.x = min(max(origin.x, screen.minX + 4), screen.maxX - size.width - 4)
        }
        listPanel.show(frame: NSRect(origin: origin, size: size), over: window)
    }

    private func prefixWidth(_ prefix: String) -> CGFloat {
        (prefix as NSString).size(withAttributes: [.font: EditorTheme.font]).width
    }

    private func hideList() {
        listPanel.hide()
    }

    private func showHint(_ signature: CallSignature, line: Int) {
        guard let view = textView, let window = view.window, let caret = view.caretScreenRect() else { return }
        hintHost.rootView = SignatureHintView(signature: signature)
        let fitting = hintHost.fittingSize
        let size = NSSize(width: min(max(fitting.width, 120), 560), height: max(fitting.height, 28))
        var origin = NSPoint(x: caret.minX - 16, y: caret.maxY + 6)
        if let screen = window.screen?.visibleFrame {
            if origin.y + size.height > screen.maxY { origin.y = caret.minY - size.height - 6 }
            origin.x = min(max(origin.x, screen.minX + 4), screen.maxX - size.width - 4)
        }
        hintLine = line
        hintPanel.show(frame: NSRect(origin: origin, size: size), over: window)
    }

    private func hideHint() {
        hintLine = nil
        hintPanel.hide()
    }

    // MARK: Text helpers

    private func isIdentifier(_ c: UInt16) -> Bool {
        (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c == 95 || c >= 0x80
    }

    private func wordStart(before location: Int, in text: NSString) -> Int {
        var i = min(location, text.length)
        while i > 0, isIdentifier(text.character(at: i - 1)) { i -= 1 }
        return i
    }

    private func line(at location: Int, in text: NSString) -> Int {
        position(of: location, in: text).0
    }

    /// 1-based line and 0-based column in Unicode scalars, as Jedi counts.
    private func position(of location: Int, in text: NSString) -> (Int, Int) {
        let end = min(location, text.length)
        let before = text.substring(to: end)
        var line = 1
        var lineStart = before.startIndex
        for i in before.indices where before[i] == "\n" {
            line += 1
            lineStart = before.index(after: i)
        }
        return (line, before[lineStart...].unicodeScalars.count)
    }

    private func insideStringOrComment(at location: Int, in code: String) -> Bool {
        PythonTokenizer.tokenize(code).contains { token in
            (token.kind == .string || token.kind == .comment)
                && token.range.location < location && location <= NSMaxRange(token.range)
                && !(token.kind == .string && location == NSMaxRange(token.range) && isClosedString(token, in: code))
        }
    }

    /// Just after a closing quote is outside the string.
    private func isClosedString(_ token: PythonTokenizer.Token, in code: String) -> Bool {
        let ns = code as NSString
        guard token.range.length >= 2 else { return false }
        let last = ns.character(at: NSMaxRange(token.range) - 1)
        return last == 34 || last == 39
    }
}

// MARK: - Panel

/// A borderless, non-activating panel that floats over the editor, with a
/// glass (macOS 26) or popover-material background.
final class FloatingPanel: NSPanel {
    init(content: NSView, cornerRadius: CGFloat) {
        super.init(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel],
                   backing: .buffered, defer: true)
        isFloatingPanel = true
        level = .popUpMenu
        hidesOnDeactivate = true
        backgroundColor = .clear
        isOpaque = false
        hasShadow = true
        isReleasedWhenClosed = false
        animationBehavior = .none
        install(content, cornerRadius: cornerRadius)
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    private func install(_ content: NSView, cornerRadius: CGFloat) {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            let glass = NSGlassEffectView()
            glass.cornerRadius = cornerRadius
            glass.contentView = content
            contentView = glass
            return
        }
        #endif
        let effect = NSVisualEffectView()
        effect.material = .popover
        effect.blendingMode = .behindWindow
        effect.state = .active
        effect.wantsLayer = true
        effect.layer?.cornerRadius = cornerRadius
        effect.layer?.cornerCurve = .continuous
        effect.layer?.masksToBounds = true
        content.translatesAutoresizingMaskIntoConstraints = false
        effect.addSubview(content)
        NSLayoutConstraint.activate([
            content.leadingAnchor.constraint(equalTo: effect.leadingAnchor),
            content.trailingAnchor.constraint(equalTo: effect.trailingAnchor),
            content.topAnchor.constraint(equalTo: effect.topAnchor),
            content.bottomAnchor.constraint(equalTo: effect.bottomAnchor),
        ])
        contentView = effect
    }

    func show(frame: NSRect, over window: NSWindow) {
        let appearing = !isVisible
        setFrame(frame, display: true)
        if parent !== window {
            parent?.removeChildWindow(self)
            window.addChildWindow(self, ordered: .above)
        }
        if appearing {
            alphaValue = 0
            orderFront(nil)
            NSAnimationContext.runAnimationGroup { context in
                context.duration = 0.12
                animator().alphaValue = 1
            }
        }
    }

    func hide() {
        guard isVisible else { return }
        parent?.removeChildWindow(self)
        orderOut(nil)
    }
}

// MARK: - Views

struct CompletionListView: View {
    let items: [Completion]
    let selected: Int
    let prefix: String
    let onPick: (Int) -> Void

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    LazyVStack(spacing: 0) {
                        ForEach(Array(items.enumerated()), id: \.offset) { i, item in
                            CompletionRow(item: item, prefix: prefix, isSelected: i == selected)
                                .id(i)
                                .contentShape(Rectangle())
                                .onTapGesture { onPick(i) }
                        }
                    }
                    .padding(6)
                }
                .scrollIndicators(.never)
                .onAppear { proxy.scrollTo(selected) }
                .onChange(of: selected) { _, i in proxy.scrollTo(i) }
            }
            if items.indices.contains(selected), items[selected].signature != nil || items[selected].doc != nil {
                Divider().opacity(0.6)
                CompletionDetail(item: items[selected])
                    .frame(height: CompletionController.detailHeight, alignment: .topLeading)
            }
        }
        .frame(width: CompletionController.listWidth)
    }
}

private struct CompletionRow: View {
    let item: Completion
    let prefix: String
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 8) {
            KindBadge(type: item.type)
            label
                .lineLimit(1)
            Spacer(minLength: 8)
            Text(item.type)
                .font(.system(size: 10.5))
                .foregroundStyle(isSelected ? AnyShapeStyle(Color.white.opacity(0.8)) : AnyShapeStyle(HierarchicalShapeStyle.tertiary))
        }
        .padding(.horizontal, 6)
        .frame(height: CompletionController.rowHeight)
        .background {
            if isSelected {
                RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Theme.accent)
            }
        }
    }

    /// The name, with the typed part in bold.
    private var label: Text {
        let font = Font.system(size: 12.5, design: .monospaced)
        let color: Color = isSelected ? .white : .primary
        if !prefix.isEmpty, item.name.lowercased().hasPrefix(prefix.lowercased()) {
            let head = String(item.name.prefix(prefix.count))
            let tail = String(item.name.dropFirst(prefix.count))
            return Text(head).font(font.weight(.bold)).foregroundColor(color)
                + Text(tail).font(font).foregroundColor(color.opacity(0.85))
        }
        return Text(item.name).font(font).foregroundColor(color)
    }
}

private struct KindBadge: View {
    let type: String

    private var info: (String, Color) {
        switch type {
        case "function": ("ƒ", .purple)
        case "class": ("C", .orange)
        case "module": ("M", .blue)
        case "keyword": ("K", .pink)
        case "param": ("P", .teal)
        case "property": ("P", .indigo)
        case "path": ("/", .gray)
        default: ("V", .green)
        }
    }

    var body: some View {
        Text(info.0)
            .font(.system(size: 10, weight: .bold, design: .rounded))
            .foregroundStyle(.white)
            .frame(width: 16, height: 16)
            .background(info.1.gradient, in: RoundedRectangle(cornerRadius: 4, style: .continuous))
    }
}

private struct CompletionDetail: View {
    let item: Completion

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let signature = item.signature {
                Text(EditorTheme.attributed(signature, size: 11.5))
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            if let doc = item.doc {
                Text(doc)
                    .font(.system(size: 11.5))
                    .foregroundStyle(.secondary)
                    .lineLimit(item.signature == nil ? 4 : 3)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }
}

struct SignatureHintView: View {
    let signature: CallSignature?

    var body: some View {
        if let signature {
            VStack(alignment: .leading, spacing: 3) {
                Text(formatted(signature))
                    .lineLimit(2)
                if let doc = signature.doc {
                    Text(doc)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .frame(maxWidth: 560, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
        } else {
            Color.clear.frame(width: 1, height: 1)
        }
    }

    /// `name(a, b, c)` with the parameter at the cursor emphasised.
    private func formatted(_ sig: CallSignature) -> AttributedString {
        typealias UI = AttributeScopes.SwiftUIAttributes
        let mono = Font.system(size: 12, design: .monospaced)
        func piece(_ s: String, bold: Bool = false, color: Color = .secondary) -> AttributedString {
            var a = AttributedString(s)
            a[UI.FontAttribute.self] = bold ? mono.weight(.bold) : mono
            a[UI.ForegroundColorAttribute.self] = color
            return a
        }
        var out = piece(sig.name, color: .primary)
        out += piece("(")
        for (i, param) in sig.params.enumerated() {
            if i > 0 { out += piece(", ") }
            out += i == sig.index ? piece(param, bold: true, color: Theme.accent) : piece(param)
        }
        out += piece(")")
        return out
    }
}
