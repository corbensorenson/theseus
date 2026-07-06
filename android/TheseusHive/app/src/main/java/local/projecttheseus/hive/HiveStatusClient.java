package local.projecttheseus.hive;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

final class HiveStatusClient {
    HiveStatus fetch(URL url, String token) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setConnectTimeout(8000);
        connection.setReadTimeout(8000);
        connection.setRequestMethod("GET");
        connection.setRequestProperty("X-Theseus-Hive-Secret", token);
        int code = connection.getResponseCode();
        if (code < 200 || code >= 300) {
            return HiveStatus.rejected();
        }
        String body = read(connection);
        JSONObject object = new JSONObject(body);
        JSONObject hive = object.optJSONObject("hive");
        if (hive == null) {
            return HiveStatus.offline("Hive returned an unexpected operator response.");
        }
        JSONObject local = hive.optJSONObject("local_node");
        String nodeName = local == null ? "Theseus Hive" : local.optString("node_name", "Theseus Hive");
        int peerCount = hive.optInt("peer_count", 0);
        HiveStatus.State state = peerCount > 0 ? HiveStatus.State.CONNECTED : HiveStatus.State.RUNNING;
        String title = state == HiveStatus.State.CONNECTED ? "Connected" : "Running Locally";
        String subtitle = peerCount > 0
            ? nodeName + " sees " + peerCount + " peer(s)."
            : nodeName + " is online.";
        return new HiveStatus(state, title, subtitle, peerCount);
    }

    private String read(HttpURLConnection connection) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
        StringBuilder out = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            out.append(line).append('\n');
        }
        reader.close();
        return out.toString();
    }
}
