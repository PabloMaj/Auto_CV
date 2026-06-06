from pathlib import Path


def state_to_json(state: dict) -> dict:
    from pydantic import BaseModel

    def _convert(v):
        if isinstance(v, BaseModel):
            return _convert(v.model_dump())
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {k: _convert(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_convert(i) for i in v]
        return v

    return {k: _convert(v) for k, v in state.items()}
