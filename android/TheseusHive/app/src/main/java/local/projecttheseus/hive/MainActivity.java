package local.projecttheseus.hive;

import android.app.Activity;
import android.app.AlertDialog;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONException;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int BG = Color.rgb(16, 18, 20);
    private static final int PANEL = Color.rgb(23, 27, 32);
    private static final int PANEL_2 = Color.rgb(31, 37, 43);
    private static final int TEXT = Color.rgb(232, 237, 242);
    private static final int MUTED = Color.rgb(152, 164, 175);
    private static final int GOOD = Color.rgb(57, 217, 138);
    private static final int WARN = Color.rgb(246, 196, 83);
    private static final int BAD = Color.rgb(255, 107, 107);
    private static final int ACCENT = Color.rgb(78, 155, 216);

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final HiveStatusClient statusClient = new HiveStatusClient();

    private HiveSettings settings;
    private TextView title;
    private TextView subtitle;
    private TextView peers;
    private View dot;
    private FrameLayout body;
    private WebView webView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        settings = new HiveSettings(this, getIntent());
        buildShell();
        showConfiguredSurface();
        refreshStatus();
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }

    private void buildShell() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        setContentView(root);

        LinearLayout strip = new LinearLayout(this);
        strip.setOrientation(LinearLayout.HORIZONTAL);
        strip.setGravity(Gravity.CENTER_VERTICAL);
        strip.setPadding(dp(14), dp(10), dp(10), dp(10));
        strip.setBackgroundColor(PANEL);
        root.addView(strip, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(66)));

        dot = new View(this);
        LinearLayout.LayoutParams dotParams = new LinearLayout.LayoutParams(dp(10), dp(10));
        dotParams.setMargins(0, 0, dp(10), 0);
        strip.addView(dot, dotParams);

        LinearLayout labels = new LinearLayout(this);
        labels.setOrientation(LinearLayout.VERTICAL);
        strip.addView(labels, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

        title = new TextView(this);
        title.setTextColor(TEXT);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        title.setTextSize(15);
        labels.addView(title);

        subtitle = new TextView(this);
        subtitle.setTextColor(MUTED);
        subtitle.setTextSize(12);
        subtitle.setSingleLine(true);
        labels.addView(subtitle);

        peers = new TextView(this);
        peers.setTextColor(TEXT);
        peers.setTypeface(Typeface.MONOSPACE, Typeface.BOLD);
        peers.setGravity(Gravity.CENTER);
        strip.addView(peers, new LinearLayout.LayoutParams(dp(50), ViewGroup.LayoutParams.WRAP_CONTENT));

        Button settingsButton = compactButton("Settings");
        settingsButton.setOnClickListener(v -> showSettingsDialog());
        strip.addView(settingsButton);

        Button refreshButton = compactButton("Refresh");
        refreshButton.setOnClickListener(v -> {
            showConfiguredSurface();
            refreshStatus();
        });
        strip.addView(refreshButton);

        body = new FrameLayout(this);
        root.addView(body, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
    }

    private Button compactButton(String label) {
        Button button = new Button(this);
        button.setText(label);
        button.setAllCaps(false);
        button.setTextColor(TEXT);
        button.setBackgroundColor(PANEL_2);
        button.setPadding(dp(8), 0, dp(8), 0);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(42));
        params.setMargins(dp(6), 0, 0, 0);
        button.setLayoutParams(params);
        return button;
    }

    private void showConfiguredSurface() {
        body.removeAllViews();
        if (!settings.isConfigured()) {
            body.addView(onboardingView());
            applyStatus(HiveStatus.notConfigured());
            return;
        }
        webView = new WebView(this);
        WebSettings webSettings = webView.getSettings();
        webSettings.setJavaScriptEnabled(true);
        webSettings.setDomStorageEnabled(true);
        webSettings.setDatabaseEnabled(true);
        webSettings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        webView.setWebViewClient(new WebViewClient());
        body.addView(webView, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        webView.loadUrl(settings.authenticatedMobileUrl());
    }

    private View onboardingView() {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setGravity(Gravity.CENTER);
        box.setPadding(dp(26), dp(26), dp(26), dp(26));
        box.setBackgroundColor(BG);

        ImageView icon = new ImageView(this);
        icon.setImageResource(getResources().getIdentifier("theseus_hex", "drawable", getPackageName()));
        box.addView(icon, new LinearLayout.LayoutParams(dp(82), dp(82)));

        TextView headline = new TextView(this);
        headline.setText("Connect to Theseus Hive");
        headline.setTextColor(TEXT);
        headline.setTextSize(22);
        headline.setTypeface(Typeface.DEFAULT_BOLD);
        headline.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams headlineParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        headlineParams.setMargins(0, dp(18), 0, dp(8));
        box.addView(headline, headlineParams);

        TextView copy = new TextView(this);
        copy.setText("Add a trusted node URL and invite token to use the private operator from this Android device.");
        copy.setTextColor(MUTED);
        copy.setTextSize(15);
        copy.setGravity(Gravity.CENTER);
        box.addView(copy);

        Button setup = compactButton("Set Up Hive Access");
        setup.setBackgroundColor(ACCENT);
        setup.setOnClickListener(v -> showSettingsDialog());
        LinearLayout.LayoutParams buttonParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(48));
        buttonParams.setMargins(0, dp(22), 0, 0);
        box.addView(setup, buttonParams);
        return box;
    }

    private void showSettingsDialog() {
        LinearLayout form = new LinearLayout(this);
        form.setOrientation(LinearLayout.VERTICAL);
        form.setPadding(dp(18), dp(10), dp(18), 0);

        EditText nodeUrl = input("http://10.0.2.2:8791", InputType.TYPE_TEXT_VARIATION_URI);
        nodeUrl.setText(settings.getNodeUrl());
        form.addView(label("Hive node URL"));
        form.addView(nodeUrl);

        EditText token = input("Hive invite token", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        token.setText(settings.getToken());
        form.addView(label("Invite token"));
        form.addView(token);

        EditText invite = input("Paste invite JSON", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        invite.setMinLines(5);
        invite.setGravity(Gravity.TOP | Gravity.START);
        form.addView(label("Import invite JSON"));
        form.addView(invite);

        Button parse = compactButton("Parse Invite JSON");
        parse.setOnClickListener(v -> {
            try {
                HiveInvite parsed = HiveInvite.parse(invite.getText().toString());
                nodeUrl.setText(parsed.coordinatorUrl);
                token.setText(parsed.joinToken);
                toast("Invite loaded for " + parsed.hiveId + ".");
            } catch (JSONException error) {
                toast(error.getMessage());
            }
        });
        form.addView(parse, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(44)));

        ScrollView scroll = new ScrollView(this);
        scroll.addView(form);

        new AlertDialog.Builder(this)
            .setTitle("Hive Settings")
            .setView(scroll)
            .setPositiveButton("Save", (dialog, which) -> {
                try {
                    settings.save(nodeUrl.getText().toString(), token.getText().toString());
                    showConfiguredSurface();
                    refreshStatus();
                } catch (Exception error) {
                    toast(error.getMessage());
                    showSettingsDialog();
                }
            })
            .setNegativeButton("Cancel", null)
            .setNeutralButton("Clear", (dialog, which) -> {
                settings.clear();
                showConfiguredSurface();
                applyStatus(HiveStatus.notConfigured());
            })
            .show();
    }

    private TextView label(String text) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextColor(MUTED);
        view.setTextSize(12);
        view.setPadding(0, dp(12), 0, dp(4));
        return view;
    }

    private EditText input(String hint, int inputType) {
        EditText edit = new EditText(this);
        edit.setHint(hint);
        edit.setInputType(inputType);
        edit.setSingleLine((inputType & InputType.TYPE_TEXT_FLAG_MULTI_LINE) == 0);
        edit.setTextColor(TEXT);
        edit.setHintTextColor(MUTED);
        edit.setBackgroundColor(PANEL_2);
        edit.setPadding(dp(10), dp(8), dp(10), dp(8));
        return edit;
    }

    private void refreshStatus() {
        if (!settings.isConfigured()) {
            applyStatus(HiveStatus.notConfigured());
            return;
        }
        executor.execute(() -> {
            HiveStatus result;
            try {
                result = statusClient.fetch(settings.operatorStatusUrl(), settings.getToken());
            } catch (Exception error) {
                result = HiveStatus.offline(error.getMessage());
            }
            HiveStatus finalResult = result;
            main.post(() -> applyStatus(finalResult));
        });
    }

    private void applyStatus(HiveStatus status) {
        title.setText(status.title);
        subtitle.setText(status.subtitle);
        peers.setText(status.peerCount + "\npeers");
        int color;
        switch (status.state) {
            case CONNECTED:
                color = GOOD;
                break;
            case RUNNING:
                color = WARN;
                break;
            default:
                color = BAD;
                break;
        }
        dot.setBackgroundColor(color);
    }

    private void toast(String text) {
        Toast.makeText(this, text == null ? "" : text, Toast.LENGTH_LONG).show();
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
