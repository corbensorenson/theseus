import Foundation
import NaturalLanguage

guard CommandLine.arguments.count == 2 else {
    fputs("expected exactly one title\n", stderr)
    exit(2)
}

let title = CommandLine.arguments[1]
let recognizer = NLLanguageRecognizer()
recognizer.processString(title)
let dominant = recognizer.dominantLanguage?.rawValue ?? "unknown"
let hypotheses = recognizer.languageHypotheses(withMaximum: 3)
    .map { ["language": $0.key.rawValue, "probability": $0.value] }
    .sorted { left, right in
        let lp = left["probability"] as? Double ?? 0.0
        let rp = right["probability"] as? Double ?? 0.0
        return lp > rp
    }
let payload: [String: Any] = [
    "dominant_language": dominant,
    "hypotheses": hypotheses,
]
let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
