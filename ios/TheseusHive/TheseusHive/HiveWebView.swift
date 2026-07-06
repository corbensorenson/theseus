import SwiftUI
@preconcurrency import WebKit

struct HiveWebView: UIViewRepresentable {
    let url: URL
    let token: String
    let reloadID: UUID

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.allowsBackForwardNavigationGestures = true
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        guard context.coordinator.reloadID != reloadID || context.coordinator.url != url || context.coordinator.token != token else {
            return
        }
        context.coordinator.reloadID = reloadID
        context.coordinator.url = url
        context.coordinator.token = token
        webView.load(URLRequest(url: authenticatedURL(base: url, token: token)))
    }

    private func authenticatedURL(base: URL, token: String) -> URL {
        guard !token.isEmpty, var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else {
            return base
        }
        var queryItems = components.queryItems ?? []
        queryItems.removeAll { $0.name == "token" || $0.name == "t" || $0.name == "native" }
        components.queryItems = queryItems.isEmpty ? nil : queryItems
        let fragmentItems = [
            URLQueryItem(name: "token", value: token),
            URLQueryItem(name: "native", value: "1")
        ]
        var fragment = URLComponents()
        fragment.queryItems = fragmentItems
        components.fragment = fragment.percentEncodedQuery
        return components.url ?? base
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var reloadID: UUID?
        var url: URL?
        var token = ""

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            guard let targetURL = navigationAction.request.url else {
                decisionHandler(.cancel)
                return
            }
            if targetURL.scheme == "http" || targetURL.scheme == "https" {
                decisionHandler(.allow)
            } else {
                UIApplication.shared.open(targetURL)
                decisionHandler(.cancel)
            }
        }
    }
}
