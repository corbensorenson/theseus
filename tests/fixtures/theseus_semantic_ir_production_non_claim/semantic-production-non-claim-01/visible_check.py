from sample import classify


def main() -> None:
    assert classify("warn") == "warning", "SEMANTIC_PRODUCTION_VISIBLE_WARN_MISSING"
    assert classify("info") == "information", "SEMANTIC_PRODUCTION_VISIBLE_INFO_REGRESSED"
    assert classify("error") == "error", "SEMANTIC_PRODUCTION_VISIBLE_ERROR_REGRESSED"
    assert classify("other") == "unknown", "SEMANTIC_PRODUCTION_VISIBLE_FALLBACK_REGRESSED"


if __name__ == "__main__":
    main()
