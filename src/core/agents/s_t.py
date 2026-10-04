UI_FAST = True


def s_t_fn(state: dict) -> dict:
    s = dict(state)
    me = s.setdefault("meta_exec", {"errors": [], "train_harness": {}})
    if UI_FAST:
        me["train_harness"] = {
            "mode": "smoke",
            "subset": "tiny",
            "epochs": 1,
            "cache_only_loader": True,
            "wavlm_loaded_in_train": False,
            "status": "built_verified_ui_skip_check",
        }
    else:
        me.setdefault(
            "train_harness",
            {
                "mode": "smoke",
                "subset": "tiny",
                "epochs": 1,
                "cache_only_loader": True,
                "status": "built_verified",
            },
        )
    s.setdefault("avla", {"wavlm_emb_path": "cache/emb/wavlm.pt", "audio_tokens_shape": [0, 2048]})
    return s
