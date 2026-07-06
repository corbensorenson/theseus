package local.projecttheseus.hive;

final class HiveStatus {
    enum State {
        OFFLINE,
        RUNNING,
        CONNECTED
    }

    final State state;
    final String title;
    final String subtitle;
    final int peerCount;

    HiveStatus(State state, String title, String subtitle, int peerCount) {
        this.state = state;
        this.title = title;
        this.subtitle = subtitle;
        this.peerCount = peerCount;
    }

    static HiveStatus notConfigured() {
        return new HiveStatus(State.OFFLINE, "Not Connected", "Add a Hive node URL and invite token.", 0);
    }

    static HiveStatus offline(String detail) {
        return new HiveStatus(State.OFFLINE, "Hive Offline", detail == null ? "Unable to reach Hive." : detail, 0);
    }

    static HiveStatus rejected() {
        return new HiveStatus(State.OFFLINE, "Hive Rejected Request", "Check the invite token on this device.", 0);
    }
}
