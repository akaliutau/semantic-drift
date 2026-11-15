"""Presentation-tolerant JSON with no content correction or draft selection."""

import json
from .storage import strict_json


def final_object(text):
    text=text.strip()
    if text.endswith('```'):
        end=text.rfind('```');start=text.rfind('```',0,end)
        if start>=0:
            body=text[start+3:end].lstrip()
            if body.startswith('json'):body=body[4:].lstrip()
            text=body
    try:
        obj=strict_json(text)
        if not isinstance(obj,dict):raise ValueError('One JSON object required')
        return obj
    except ValueError:pass
    # Only an object ending the response may count. Earlier drafts never count.
    decoder=json.JSONDecoder()
    for start,c in enumerate(text):
        if c!='{':continue
        try:_,end=decoder.raw_decode(text[start:])
        except ValueError:continue
        if not text[start+end:].strip():
            obj=strict_json(text[start:])
            if isinstance(obj,dict):return obj
    raise ValueError('No complete terminal JSON object')
