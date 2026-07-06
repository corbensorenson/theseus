package local.projecttheseus.hive;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;

import java.net.MalformedURLException;
import java.net.URL;

final class HiveSettings {
    private static final String PREFS = "theseus_hive_settings";
    private static final String NODE_URL = "node_url";

    private final SharedPreferences prefs;
    private final SecureTokenStore tokenStore;
    private String nodeUrl;
    private String token;

    HiveSettings(Context context, Intent launchIntent) {
        prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        tokenStore = new SecureTokenStore(context);
        nodeUrl = prefs.getString(NODE_URL, "");
        token = tokenStore.read();
        applyLaunchOverrides(launchIntent);
    }

    boolean isConfigured() {
        return !nodeUrl.isEmpty() && !token.isEmpty();
    }

    String getNodeUrl() {
        return nodeUrl;
    }

    String getToken() {
        return token;
    }

    URL operatorStatusUrl() throws MalformedURLException {
        return new URL(nodeUrl + "/api/hive/operator/status");
    }

    String authenticatedMobileUrl() {
        String separator = nodeUrl.contains("?") ? "&" : "?";
        return nodeUrl + "/mobile" + separator + "native=1&token=" + urlEncode(token);
    }

    void save(String rawNodeUrl, String rawToken) throws Exception {
        String normalized = normalizeBaseUrl(rawNodeUrl);
        String cleanToken = rawToken == null ? "" : rawToken.trim();
        if (normalized.isEmpty()) {
            throw new IllegalArgumentException("Enter a valid Hive node URL, for example http://10.0.2.2:8791.");
        }
        if (cleanToken.isEmpty()) {
            throw new IllegalArgumentException("Enter the private Hive invite token.");
        }
        tokenStore.save(cleanToken);
        prefs.edit().putString(NODE_URL, normalized).apply();
        nodeUrl = normalized;
        token = cleanToken;
    }

    void clear() {
        prefs.edit().remove(NODE_URL).apply();
        tokenStore.clear();
        nodeUrl = "";
        token = "";
    }

    private void applyLaunchOverrides(Intent intent) {
        if (intent == null) {
            return;
        }
        String extraUrl = intent.getStringExtra("theseus_node_url");
        String extraToken = intent.getStringExtra("theseus_token");
        if (extraUrl == null || extraToken == null) {
            return;
        }
        try {
            save(extraUrl, extraToken);
        } catch (Exception ignored) {
            // Ignore invalid launch extras; the settings screen will explain the required shape.
        }
    }

    static String normalizeBaseUrl(String raw) {
        String text = raw == null ? "" : raw.trim();
        if (text.isEmpty()) {
            return "";
        }
        if (!text.contains("://")) {
            text = "http://" + text;
        }
        while (text.endsWith("/")) {
            text = text.substring(0, text.length() - 1);
        }
        try {
            URL url = new URL(text);
            if (url.getHost() == null || url.getHost().isEmpty()) {
                return "";
            }
            return text;
        } catch (MalformedURLException ignored) {
            return "";
        }
    }

    private static String urlEncode(String value) {
        try {
            return java.net.URLEncoder.encode(value, "UTF-8");
        } catch (Exception ignored) {
            return "";
        }
    }
}
