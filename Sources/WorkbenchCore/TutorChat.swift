import Foundation

/// The tutor conversation as the panel shows it, and the history sent back
/// with each question. Pure state, so it's testable without the sidecar.
public struct TutorChat: Equatable, Sendable {
    public struct Message: Identifiable, Equatable, Sendable {
        public var id: Int
        public var role: TutorTurn.Role
        public var text: String
        /// Shown under the message, e.g. why an answer stopped.
        public var note: String?
        public var isError = false
        /// Errors and stopped answers aren't replayed to the model.
        public var inHistory = true
    }

    public private(set) var messages: [Message] = []
    /// The sidecar request whose answer is streaming in, if any.
    public private(set) var activeRequest: String?
    private var nextID = 0

    public init() {}

    public var isAnswering: Bool { activeRequest != nil }

    /// Everything said so far that should go back to the model.
    public var history: [TutorTurn] {
        messages.filter { $0.inHistory && !$0.text.isEmpty }.map { TutorTurn(role: $0.role, text: $0.text) }
    }

    /// Adds the learner's question and an empty answer to stream into.
    /// Call `history` first: it's the history *before* this question.
    public mutating func ask(_ question: String, request: String) {
        append(.user, question)
        append(.assistant, "")
        activeRequest = request
    }

    public mutating func receive(delta text: String, request: String) {
        guard request == activeRequest, let i = messages.indices.last else { return }
        messages[i].text += text
    }

    /// `stopReason` from `tutor_done`.
    public mutating func finish(request: String, stopReason: String?, note: String?) {
        guard request == activeRequest, let i = messages.indices.last else { return }
        switch stopReason {
        case "cancelled":
            messages[i].note = "Stopped."
            messages[i].inHistory = false
        case "refusal":
            messages[i].note = note ?? "The model declined to answer."
            messages[i].inHistory = false
        case "max_tokens":
            messages[i].note = "The answer hit the length limit."
        default:
            messages[i].note = note
        }
    }

    public mutating func fail(request: String, message: String) {
        guard request == activeRequest, let i = messages.indices.last else { return }
        messages[i].isError = true
        messages[i].inHistory = false
        messages[i].text = message
        // The question stays visible but isn't replayed: ask again to retry.
        if i > 0 { messages[i - 1].inHistory = false }
    }

    /// The request is over (its `done` event arrived).
    public mutating func end(request: String) {
        guard request == activeRequest else { return }
        if let i = messages.indices.last, messages[i].role == .assistant, messages[i].text.isEmpty,
           !messages[i].isError {
            messages[i].inHistory = false
            if messages[i].note == nil { messages[i].note = "No answer." }
        }
        activeRequest = nil
    }

    public mutating func clear() {
        messages.removeAll()
        activeRequest = nil
    }

    private mutating func append(_ role: TutorTurn.Role, _ text: String) {
        nextID += 1
        messages.append(Message(id: nextID, role: role, text: text))
    }
}
