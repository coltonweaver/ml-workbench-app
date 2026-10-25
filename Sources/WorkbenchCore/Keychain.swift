import Foundation
import Security

/// One secret in the login keychain, stored as a generic password.
public struct KeychainItem: Sendable {
    public var service: String
    public var account: String

    /// Where the tutor's Claude API key lives.
    public static let apiKey = KeychainItem(service: "ML Workbench", account: "anthropic-api-key")

    public init(service: String, account: String) {
        self.service = service
        self.account = account
    }

    public struct Failure: Error, LocalizedError, Equatable {
        public var status: OSStatus
        public var errorDescription: String? {
            let message = SecCopyErrorMessageString(status, nil) as String? ?? "unknown error"
            return "Keychain error \(status): \(message)"
        }
    }

    private var query: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }

    public func read() throws -> String? {
        var q = query
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(q as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = result as? Data else { throw Failure(status: status) }
        return String(decoding: data, as: UTF8.self)
    }

    public func save(_ value: String) throws {
        let data = Data(value.utf8)
        let update = SecItemUpdate(query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if update == errSecSuccess { return }
        guard update == errSecItemNotFound else { throw Failure(status: update) }
        var q = query
        q[kSecValueData as String] = data
        q[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlocked
        let status = SecItemAdd(q as CFDictionary, nil)
        guard status == errSecSuccess else { throw Failure(status: status) }
    }

    public func delete() throws {
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw Failure(status: status) }
    }
}
