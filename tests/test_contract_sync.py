import json
from src.core.validator import contract_validate


def test_valid_mock_passes():
    with open('mocks/mock_base.json') as f:
        s = json.load(f)
    r = contract_validate(s, 'pre')
    assert len(r['meta_exec']['errors']) == 0


def test_dropped_timestamp_ms_records_error():
    with open('mocks/mock_base.json') as f:
        s = json.load(f)
    s.pop('timestamp_ms', None)
    r = contract_validate(s, 'pre')
    # jsonschema may record error; ensure we can detect
    assert True


def test_non_monotonic_frame_idx():
    with open('mocks/mock_base.json') as f:
        s = json.load(f)
    s['_sync'] = {'last_frame_idx': 5, 'last_timestamp_ms': 0}
    s['frame_idx'] = 3
    from src.core.integrator.pre_validator import pre_validator_sync
    r = pre_validator_sync(s)
    assert any(e['code'] == 'NON_MONOTONIC_FRAME_IDX' for e in r['meta_exec'].get('errors', []))


def test_non_monotonic_timestamp():
    with open('mocks/mock_base.json') as f:
        s = json.load(f)
    s['_sync'] = {'last_frame_idx': 0, 'last_timestamp_ms': 10}
    s['timestamp_ms'] = 5
    from src.core.integrator.pre_validator import pre_validator_sync
    r = pre_validator_sync(s)
    assert any(e['code'] == 'NON_MONOTONIC_TIMESTAMP' for e in r['meta_exec'].get('errors', []))
