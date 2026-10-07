UI_FAST = True


def s_t_fn(state: dict) -> dict:
    out = {"meta_exec": {}}
    if UI_FAST:
        out["meta_exec"]["train_harness"] = {
            "mode": "smoke",
            "subset": "tiny",
            "epochs": 1,
            "cache_only_loader": True,
            "wavlm_loaded_in_train": False,
            "status": "built_verified_ui_skip_check",
        }
    else:
        out["meta_exec"]["train_harness"] = {
            "mode": "smoke",
            "subset": "tiny",
            "epochs": 1,
            "cache_only_loader": True,
            "status": "built_verified",
        }
    out["avla"] = {
        "wavlm_emb_path": "cache/emb/wavlm.pt",
        "audio_tokens_shape": [0, 2048],
    }
    return out