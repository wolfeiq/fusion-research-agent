"""Mechanical citation transport for an already captured experiment snapshot.

This reads JSON keys and indices, never evaluates a path or validates a claim's
meaning. The normal report verifier and semantic judge remain required.
"""

from __future__ import annotations

import json


def _reject_constant(_value):
    raise ValueError("Experiment JSON contains a non-finite number")


def _path_hint(source, path, value):
    locator = "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in path) if path else ""
    hint = {"source_id": source["id"], "path": list(path), "locator": locator}
    if isinstance(value, dict):
        keys = list(value)
        shown = [key for key in keys[:32] if len(key) <= 128 and not any(ord(char) < 32 for char in key)]
        hint.update(value_type="object", child_keys=shown, omitted_children=len(keys) - len(shown))
    elif isinstance(value, list):
        indices = list(range(min(len(value), 32)))
        hint.update(value_type="array", child_indices=indices, omitted_children=len(value) - len(indices))
    else:
        hint.update(value_type="scalar", child_keys=[], omitted_children=0)
    return hint


def _path_error(message, source, path, value):
    error = ValueError(message + ". Use citation_path_hint for the last resolved value and its available children.")
    error.citation_path_hint = _path_hint(source, path, value)
    return error


def cite_experiment_value(source, path):
    if not isinstance(source, dict) or source.get("kind") != "experiment":
        raise ValueError("Choose an already-read experiment JSON source")
    if not isinstance(path, list) or not 1 <= len(path) <= 32:
        raise ValueError("path must contain 1 to 32 string keys or integer indices")
    for component in path:
        if isinstance(component, str):
            if len(component) > 128 or any(ord(char) < 32 for char in component):
                raise ValueError("Path keys must be at most 128 characters without controls")
        elif type(component) is not int or component < 0:
            raise ValueError("Path components must be string keys or nonnegative integer indices")
    locator = "/" + "/".join(
        str(component).replace("~", "~0").replace("/", "~1") for component in path
    )
    if len(locator) > 512:
        raise ValueError("The resulting citation pointer exceeds 512 characters")
    try:
        value = json.loads(source["content"], parse_constant=_reject_constant)
    except (TypeError, KeyError, ValueError):
        raise ValueError("The selected experiment does not contain valid finite JSON") from None
    for depth, component in enumerate(path):
        if isinstance(value, dict):
            if not isinstance(component, str) or component not in value:
                raise _path_error("A dictionary path component must be an existing string key", source, path[:depth], value)
        elif isinstance(value, list):
            if type(component) is not int or component >= len(value):
                raise _path_error("An array path component must be an in-range integer index", source, path[:depth], value)
        else:
            raise _path_error("The path continues beyond a recorded scalar value", source, path[:depth], value)
        value = value[component]
    quote = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if len(quote) > 2000:
        hint = _path_hint(source, path, value)
        error = ValueError(
            "Selected value exceeds the 2000-character quotation limit. "
            "Choose a deeper path or a smaller recorded value using citation_path_hint."
        )
        error.citation_path_hint = hint
        raise error
    return {"source_id": source["id"], "locator": locator, "quote": quote}
