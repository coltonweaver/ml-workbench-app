import Foundation

/// The `review` event: warm-ups for concepts with repeated mistakes, the
/// prediction score, and the result of the last warm-up answered.
public struct ReviewState: Decodable, Equatable, Sendable {
    public var warmups: [Warmup]
    public var score: PredictionScore
    public var result: WarmupResult?

    public init(warmups: [Warmup], score: PredictionScore, result: WarmupResult? = nil) {
        self.warmups = warmups
        self.score = score
        self.result = result
    }
}

public struct Warmup: Decodable, Equatable, Sendable, Identifiable {
    public var id: String
    public var concept: String
    public var lesson: String
    public var lessonTitle: String
    /// The lesson's hook, when the warm-up is that lesson's own prediction.
    public var context: String?
    /// "number" or "choice".
    public var kind: String
    public var prompt: String
    public var options: [String]?
    /// Failed check runs on this concept since it was last cleared.
    public var mistakes: Int
    public var lastMessage: String?

    /// "derivative" → "Derivative", "kv-cache" → "KV cache".
    public var conceptName: String {
        let words = concept.split(separator: "-").map { word in
            Self.acronyms.contains(String(word)) ? word.uppercased() : String(word)
        }
        let name = words.joined(separator: " ")
        return name.prefix(1).uppercased() + name.dropFirst()
    }

    static let acronyms: Set<String> = [
        "kv", "mlp", "gpt", "cnn", "gpu", "sft", "dpo", "ddp", "ttft", "int8", "int4", "fp16", "bf16", "gelu",
    ]

    enum CodingKeys: String, CodingKey {
        case id, concept, lesson, context, kind, prompt, options, mistakes
        case lessonTitle = "lesson_title"
        case lastMessage = "last_message"
    }
}

public struct WarmupResult: Decodable, Equatable, Sendable {
    public var id: String
    public var concept: String
    public var correct: Bool
    public var yours: String
    public var answer: String
    public var explain: String?
    public var lesson: String
    public var lessonTitle: String

    enum CodingKeys: String, CodingKey {
        case id, concept, correct, yours, answer, explain, lesson
        case lessonTitle = "lesson_title"
    }
}

public struct PredictionScore: Decodable, Equatable, Sendable {
    public struct Bucket: Decodable, Equatable, Sendable, Identifiable {
        public var confidence: Double
        public var count: Int
        public var hitRate: Double
        public var id: Double { confidence }

        enum CodingKeys: String, CodingKey {
            case confidence, count
            case hitRate = "hit_rate"
        }
    }

    public struct Point: Decodable, Equatable, Sendable {
        public var lesson: String
        public var correct: Bool
        public var confidence: Double?
        /// Running hit rate up to and including this prediction.
        public var hitRate: Double

        enum CodingKeys: String, CodingKey {
            case lesson, correct, confidence
            case hitRate = "hit_rate"
        }
    }

    public var count: Int
    public var correct: Int
    public var hitRate: Double?
    /// Predictions that came with a confidence.
    public var rated: Int
    /// Mean squared error of the stated confidence; 0 is perfect, 0.25 is
    /// always saying 50%.
    public var brier: Double?
    /// Mean confidence minus hit rate: positive means more sure than right.
    public var overconfidence: Double?
    public var buckets: [Bucket]
    public var history: [Point]

    public static let empty = PredictionScore(
        count: 0, correct: 0, hitRate: nil, rated: 0, brier: nil, overconfidence: nil,
        buckets: [], history: []
    )

    public init(count: Int, correct: Int, hitRate: Double?, rated: Int, brier: Double?,
                overconfidence: Double?, buckets: [Bucket], history: [Point]) {
        self.count = count
        self.correct = correct
        self.hitRate = hitRate
        self.rated = rated
        self.brier = brier
        self.overconfidence = overconfidence
        self.buckets = buckets
        self.history = history
    }

    /// One sentence on how well calibrated the learner is.
    public var verdict: String {
        guard count > 0 else {
            return "Your score starts once you've passed a lesson you made a prediction for."
        }
        guard let overconfidence, rated >= 5 else {
            return "Right \(correct) of \(count). Say how sure you are when you predict, and after five you'll see how well calibrated you are."
        }
        let points = Int((abs(overconfidence) * 100).rounded())
        if points <= 10 {
            return "Well calibrated: your confidence is within \(points) points of how often you're right."
        }
        return overconfidence > 0
            ? "Overconfident: your confidence runs \(points) points above how often you're right."
            : "Underconfident: you're right \(points) points more often than your confidence says."
    }

    enum CodingKeys: String, CodingKey {
        case count, correct, rated, brier, overconfidence, buckets, history
        case hitRate = "hit_rate"
    }
}
