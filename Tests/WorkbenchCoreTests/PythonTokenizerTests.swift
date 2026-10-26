import XCTest
@testable import WorkbenchCore

final class PythonTokenizerTests: XCTestCase {
    /// The source text of each token with its kind, for readable asserts.
    private func lex(_ code: String) -> [(String, PythonTokenizer.Kind)] {
        let ns = code as NSString
        return PythonTokenizer.tokenize(code).map { (ns.substring(with: $0.range), $0.kind) }
    }

    private func kinds(_ code: String, _ kind: PythonTokenizer.Kind) -> [String] {
        lex(code).filter { $0.1 == kind }.map(\.0)
    }

    func testDefinitionsKeywordsAndCalls() {
        let code = "def slope(f, x, h=1e-5):\n    return (f(x + h) - f(x - h)) / (2 * h)\n"
        XCTAssertEqual(kinds(code, .keyword), ["def", "return"])
        XCTAssertEqual(kinds(code, .definition), ["slope"])
        XCTAssertEqual(kinds(code, .call), ["f", "f"])
        XCTAssertEqual(kinds(code, .number), ["1e-5", "2"])
    }

    func testStringsAndComments() {
        let code = """
        s = 'it\\'s'  # a comment with "quotes"
        t = rb"raw\\d" + f'{x}'
        doc = \"\"\"multi
        line\"\"\"
        """
        XCTAssertEqual(kinds(code, .string), [#"'it\'s'"#, #"rb"raw\d""#, "f'{x}'", "\"\"\"multi\nline\"\"\""])
        XCTAssertEqual(kinds(code, .comment), [#"# a comment with "quotes""#])
    }

    func testUnterminatedStringStopsAtLineEnd() {
        let code = "x = 'open\ny = 1"
        XCTAssertEqual(kinds(code, .string), ["'open"])
        XCTAssertEqual(kinds(code, .number), ["1"])
    }

    func testDecoratorsBuiltinsConstantsSelfAndTypes() {
        let code = "@torch.no_grad()\nclass Value(Base):\n    def run(self):\n        return len([None, True]) + Tensor\n"
        XCTAssertEqual(kinds(code, .decorator), ["@torch.no_grad"])
        XCTAssertEqual(kinds(code, .definition), ["Value", "run"])
        XCTAssertEqual(kinds(code, .type), ["Base", "Tensor"])
        XCTAssertEqual(kinds(code, .selfParam), ["self"])
        XCTAssertEqual(kinds(code, .builtin), ["len"])
        XCTAssertEqual(kinds(code, .constant), ["None", "True"])
    }

    func testNamesStartingWithPrefixLettersAreNotStrings() {
        let code = "fmt = bias + raw"
        XCTAssertEqual(kinds(code, .string), [])
    }

    func testHexNumbersDontSwallowMinus() {
        XCTAssertEqual(kinds("0xE-1", .number), ["0xE", "1"])
    }

    func testRangesAreUTF16() {
        let code = "s = 'π' + x  # ✓"
        let tokens = PythonTokenizer.tokenize(code)
        let ns = code as NSString
        XCTAssertEqual(tokens.map { ns.substring(with: $0.range) }, ["'π'", "# ✓"])
    }
}
