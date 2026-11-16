"""Atomic, strict JSON and verifiable source snapshots."""

from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(text):
    def unique(pairs):
        obj={}
        for k,v in pairs:
            if k in obj:raise ValueError(f'Duplicate JSON key: {k}')
            obj[k]=v
        return obj
    def invalid(v):raise ValueError(f'Nonfinite JSON number: {v}')
    def finite(v):
        n=float(v)
        if not math.isfinite(n):invalid(v)
        return n
    return json.loads(text,object_pairs_hook=unique,parse_constant=invalid,parse_float=finite)


def read_json(path):return strict_json(Path(path).read_text())


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        f.write(dumps(value)+'\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        for row in rows:f.write(dumps(row)+'\n')
        f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def utc_now():return datetime.now(timezone.utc).isoformat()


@contextmanager
def file_lock(path,blocking=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def source_snapshot(output):
    root=Path(__file__).resolve().parent.parent
    paths=sorted((root/'pipeline').glob('*.py'))+[root/'pyproject.toml',root/'requirements.txt']
    hashes={str(p.relative_to(root)):file_sha256(p) for p in paths}
    for p in paths:
        dest=Path(output)/'source'/p.relative_to(root)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    return {'sha256':digest(hashes),'files':hashes}


def retry_after_seconds(value,now=None):
    if value is None:return None
    try:
        v=float(value)
        return max(0.,v) if math.isfinite(v) else None
    except (ValueError,TypeError):pass
    try:
        dt=parsedate_to_datetime(value)
        if dt.tzinfo is None:dt=dt.replace(tzinfo=timezone.utc)
        return max(0.,dt.timestamp()-(datetime.now(timezone.utc).timestamp() if now is None else now))
    except (ValueError,TypeError,OverflowError):return None
