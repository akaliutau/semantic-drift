"""Presentation-tolerant JSON with no content correction or draft selection."""

import json
from .artifacts import strict_json


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


def submission(response,tools):
    """One explicit final submission, irrespective of provider serialization."""
    if response.calls:
        if len(response.calls)!=1 or not tools or response.calls[0].get('name')!=tools[0]['name']:
            raise ValueError('Expected exactly one requested tool call')
        return final_object(response.calls[0]['arguments']),'native'
    try:return final_object(response.text),'terminal_json'
    except ValueError:
        if not tools:raise
    # Some hosted models serialize a forced function call as final message text.
    # Parse that exact final line, never execute it or inspect earlier drafts.
    import ast
    line=response.text.strip().splitlines()[-1]
    tree=ast.parse(line,mode='eval').body
    if (not isinstance(tree,ast.Call) or not isinstance(tree.func,ast.Name)
            or tree.func.id!=tools[0]['name'] or tree.args
            or any(k.arg is None for k in tree.keywords)):
        raise ValueError('No explicit terminal submission')
    pairs=[(k.arg,ast.literal_eval(k.value)) for k in tree.keywords]
    if len({k for k,v in pairs})!=len(pairs):raise ValueError('Duplicate submission field')
    return dict(pairs),'terminal_call'
