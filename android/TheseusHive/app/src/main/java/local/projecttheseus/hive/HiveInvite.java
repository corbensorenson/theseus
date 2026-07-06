package local.projecttheseus.hive;

import org.json.JSONException;
import org.json.JSONObject;

final class HiveInvite {
    final String hiveId;
    final String coordinatorUrl;
    final String joinToken;

    HiveInvite(String hiveId, String coordinatorUrl, String joinToken) {
        this.hiveId = hiveId;
        this.coordinatorUrl = coordinatorUrl;
        this.joinToken = joinToken;
    }

    static HiveInvite parse(String raw) throws JSONException {
        JSONObject object = new JSONObject(raw == null ? "" : raw);
        String hiveId = object.optString("hive_id", "");
        String coordinatorUrl = object.optString("coordinator_url", "");
        String joinToken = object.optString("join_token", "");
        if (hiveId.isEmpty() || coordinatorUrl.isEmpty() || joinToken.isEmpty()) {
            throw new JSONException("Invite JSON must include hive_id, coordinator_url, and join_token.");
        }
        return new HiveInvite(hiveId, coordinatorUrl, joinToken);
    }
}
