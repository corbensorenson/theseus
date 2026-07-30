from scripts.artifact_retention_reference import normalize_checkpoint_ref


def test_backslash_checkpoint_reference_is_rejected() -> None:
    assert normalize_checkpoint_ref(
        r"checkpoints/..\outside\weights.npz"
    ) == "", "request_contract:reject_backslash_checkpoint_reference"


def test_nul_checkpoint_reference_is_rejected() -> None:
    assert normalize_checkpoint_ref(
        "checkpoints/model\x00/weights.npz"
    ) == "", "request_contract:reject_nul_checkpoint_reference"


def test_existing_valid_and_forward_traversal_behavior_remains() -> None:
    assert normalize_checkpoint_ref(
        "  checkpoints/active/weights.npz,  "
    ) == "checkpoints/active/weights.npz"
    assert normalize_checkpoint_ref("checkpoints/../outside.npz") == ""
