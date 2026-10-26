import Foundation

/// A small, forgiving Python lexer for syntax highlighting. It works on
/// UTF-16 offsets so ranges drop straight into NSTextStorage, and never
/// fails: unfinished strings run to the end of their line (or of the file,
/// for triple quotes), which is what an editor wants mid-typing.
public enum PythonTokenizer {
    public enum Kind: Equatable, Sendable {
        case keyword
        /// True, False, None, Ellipsis.
        case constant
        /// self and cls.
        case selfParam
        case builtin
        case number
        case string
        case comment
        case decorator
        /// The name after `def` or `class`.
        case definition
        /// A CapitalizedName that isn't being defined: usually a class.
        case type
        /// A name followed by `(`.
        case call
    }

    public struct Token: Equatable, Sendable {
        public var range: NSRange
        public var kind: Kind

        public init(_ location: Int, _ length: Int, _ kind: Kind) {
            range = NSRange(location: location, length: length)
            self.kind = kind
        }
    }

    public static let keywords: Set<String> = [
        "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
        "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in",
        "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
        "with", "yield", "match", "case",
    ]

    public static let constants: Set<String> = ["True", "False", "None", "Ellipsis", "NotImplemented", "__name__", "__file__"]

    public static let builtins: Set<String> = [
        "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray", "bytes", "callable",
        "chr", "classmethod", "compile", "complex", "delattr", "dict", "dir", "divmod", "enumerate",
        "eval", "exec", "filter", "float", "format", "frozenset", "getattr", "globals", "hasattr",
        "hash", "help", "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len",
        "list", "locals", "map", "max", "memoryview", "min", "next", "object", "oct", "open", "ord",
        "pow", "print", "property", "range", "repr", "reversed", "round", "set", "setattr", "slice",
        "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
        "RuntimeError", "NotImplementedError", "StopIteration", "AssertionError", "ZeroDivisionError",
    ]

    public static func tokenize(_ text: String) -> [Token] {
        tokenize(utf16: Array(text.utf16))
    }

    public static func tokenize(utf16 s: [UInt16]) -> [Token] {
        var tokens: [Token] = []
        let n = s.count
        var i = 0
        var atLineStart = true
        /// The keyword before the current name, for `def name` / `class Name`.
        var previousWord = ""

        func at(_ j: Int) -> UInt16 { j < n ? s[j] : 0 }

        while i < n {
            let c = s[i]
            if c == nl {
                atLineStart = true
                i += 1
                continue
            }
            if c == space || c == tab || c == cr {
                i += 1
                continue
            }
            let lineStart = atLineStart
            atLineStart = false

            if c == hash {
                let start = i
                while i < n, s[i] != nl { i += 1 }
                tokens.append(Token(start, i - start, .comment))
                continue
            }

            if c == at_, lineStart {
                let start = i
                i += 1
                while i < n, isIdentifier(s[i]) || s[i] == dot { i += 1 }
                tokens.append(Token(start, i - start, .decorator))
                continue
            }

            if isQuote(c) || (isStringPrefix(c) && stringStart(s, i) != nil) {
                let start = i
                var raw = false
                while isStringPrefix(s[i]) {
                    if s[i] == r || s[i] == R { raw = true }
                    i += 1
                }
                let quote = s[i]
                let triple = at(i + 1) == quote && at(i + 2) == quote
                i += triple ? 3 : 1
                while i < n {
                    let d = s[i]
                    if d == backslash, !raw || at(i + 1) == quote {
                        i += 2
                        continue
                    }
                    if d == backslash {
                        i += 1
                        continue
                    }
                    if triple {
                        if d == quote, at(i + 1) == quote, at(i + 2) == quote {
                            i += 3
                            break
                        }
                    } else {
                        if d == nl { break }
                        if d == quote {
                            i += 1
                            break
                        }
                    }
                    i += 1
                }
                tokens.append(Token(start, min(i, n) - start, .string))
                previousWord = ""
                continue
            }

            if isDigit(c) || (c == dot && isDigit(at(i + 1))) {
                let start = i
                i += 1
                while i < n {
                    let d = s[i]
                    if isIdentifier(d) || d == dot {
                        i += 1
                    } else if (d == plus || d == minus), s[i - 1] == e || s[i - 1] == E,
                              !(s[start] == zero && (at(start + 1) == x || at(start + 1) == X)) {
                        i += 1
                    } else {
                        break
                    }
                }
                tokens.append(Token(start, i - start, .number))
                previousWord = ""
                continue
            }

            if isIdentifierStart(c) {
                let start = i
                while i < n, isIdentifier(s[i]) { i += 1 }
                let word = String(decoding: s[start..<i], as: UTF16.self)
                let kind: Kind?
                if previousWord == "def" || previousWord == "class" {
                    kind = .definition
                } else if keywords.contains(word) {
                    kind = .keyword
                } else if constants.contains(word) {
                    kind = .constant
                } else if word == "self" || word == "cls" {
                    kind = .selfParam
                } else if nextNonSpace(s, from: i) == lparen {
                    kind = builtins.contains(word) ? .builtin : .call
                } else if builtins.contains(word) {
                    kind = .builtin
                } else if let first = word.unicodeScalars.first, CharacterSet.uppercaseLetters.contains(first),
                          word.contains(where: \.isLowercase) {
                    kind = .type
                } else {
                    kind = nil
                }
                if let kind { tokens.append(Token(start, i - start, kind)) }
                previousWord = word
                continue
            }

            previousWord = ""
            i += 1
        }
        return tokens
    }

    // MARK: Character classes

    private static let nl: UInt16 = 10, cr: UInt16 = 13, space: UInt16 = 32, tab: UInt16 = 9
    private static let hash: UInt16 = 35, at_: UInt16 = 64, dot: UInt16 = 46, backslash: UInt16 = 92
    private static let plus: UInt16 = 43, minus: UInt16 = 45, lparen: UInt16 = 40
    private static let zero: UInt16 = 48, e: UInt16 = 101, E: UInt16 = 69, x: UInt16 = 120, X: UInt16 = 88
    private static let r: UInt16 = 114, R: UInt16 = 82

    private static func isQuote(_ c: UInt16) -> Bool { c == 34 || c == 39 }
    private static func isDigit(_ c: UInt16) -> Bool { c >= 48 && c <= 57 }
    private static func isIdentifierStart(_ c: UInt16) -> Bool {
        (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c == 95 || c >= 0x80
    }
    private static func isIdentifier(_ c: UInt16) -> Bool { isIdentifierStart(c) || isDigit(c) }

    /// r, b, f, u and their capitals.
    private static func isStringPrefix(_ c: UInt16) -> Bool {
        [114, 82, 98, 66, 102, 70, 117, 85].contains(c)
    }

    /// The quote's offset when a string prefix of one or two letters starts at
    /// `i` and isn't just the start of a name.
    private static func stringStart(_ s: [UInt16], _ i: Int) -> Int? {
        if i > 0, isIdentifier(s[i - 1]) { return nil }
        var j = i
        while j < s.count, j - i < 2, isStringPrefix(s[j]) { j += 1 }
        return j < s.count && isQuote(s[j]) ? j : nil
    }

    private static func nextNonSpace(_ s: [UInt16], from i: Int) -> UInt16 {
        var j = i
        while j < s.count, s[j] == space || s[j] == tab { j += 1 }
        return j < s.count ? s[j] : 0
    }
}
