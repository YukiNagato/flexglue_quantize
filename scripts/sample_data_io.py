"""Selective, CRC-checked HTTP ZIP/HDF5 downloads without changing sfm packages."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bisect
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import threading
import time
import urllib.parse
import zlib

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('FLEXGLUE_DATA_SOURCES', ROOT / 'dataset_sources')) / 'http_cache'
CACHE.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()
NETWORK_BYTES = 0


def run_curl(command, url):
    """Pass existing HF credentials through stdin, never command-line arguments."""
    config=None
    if urllib.parse.urlsplit(url).hostname=='huggingface.co':
        try:
            from huggingface_hub import get_token
            token=get_token()
        except ImportError:token=None
        if token:
            if any(c in token for c in '\r\n"\\'):raise ValueError('Invalid authentication token encoding')
            config=f'header = "Authorization: Bearer {token}"\n'.encode()
            command=command+['--config','-']
    # curl -L does not forward Authorization to other hosts without --location-trusted.
    return subprocess.run(command+[url],input=config,capture_output=True)


def fetch(url, start=None, length=None, max_bytes=64*1024**2, cache=True):
    """curl honors the workspace SOCKS proxy; require exact HTTP ranges."""
    global NETWORK_BYTES
    key = hashlib.sha256(f'{url}|{start}|{length}'.encode()).hexdigest()
    dest = CACHE / key
    if cache and dest.exists():
        return dest.read_bytes()
    query = '&' if '?' in url else '?'
    request_url = url if start is None else url + query + f'sample_range={start}-{start+length-1}'
    with tempfile.TemporaryDirectory(dir=CACHE) as directory:
        body, headers = Path(directory)/'body', Path(directory)/'headers'
        command = ['curl', '-4', '-L', '--fail', '--silent', '--show-error',
            '--connect-timeout', '20', '--max-time', '120', '--retry', '3',
            '--retry-delay', '1', '--max-filesize', str(max_bytes),
            '-D', str(headers), '-o', str(body)]
        if start is not None:
            command += ['--range', f'{start}-{start+length-1}']
        result = run_curl(command, request_url)
        if result.returncode:
            raise IOError(f'curl {result.returncode}: {result.stderr.decode()[-250:]} for {url.split("?")[0]}')
        payload = body.read_bytes()
        if start is not None:
            ranges = re.findall(r'content-range:\s*bytes (\d+)-(\d+)/(\d+)', headers.read_text(), re.I)
            if not ranges or tuple(map(int, ranges[-1][:2])) != (start, start+length-1) or len(payload) != length:
                raise IOError(f'Server returned an incorrect HTTP range: {url.split("?")[0]}')
        with _lock:
            NETWORK_BYTES += len(payload)
        if cache:
            temp = dest.with_suffix(f'.{os.getpid()}-{threading.get_ident()}.tmp')
            temp.write_bytes(payload)
            temp.replace(dest)
        return payload


def prefetch_blocks(url, starts, block_size, size, workers=8):
    """Coalesce missing adjacent HDF5 blocks; retain canonical per-block cache."""
    from concurrent.futures import ThreadPoolExecutor
    missing=[]
    for start in sorted(starts):
        length=min(block_size,size-start)
        key=hashlib.sha256(f'{url}|{start}|{length}'.encode()).hexdigest()
        if not (CACHE/key).exists():missing.append(start)
    groups=[]
    for start in missing:
        if groups and start==groups[-1][-1]+block_size and len(groups[-1])<16:groups[-1].append(start)
        else:groups.append([start])
    print('Prefetch missing',len(missing),'blocks in',len(groups),'HTTP requests',flush=True)
    def read(group):
        end=min(size,group[-1]+block_size)
        data=fetch(url,group[0],end-group[0],max_bytes=end-group[0]+1024,cache=False)
        for start in group:
            length=min(block_size,size-start)
            key=hashlib.sha256(f'{url}|{start}|{length}'.encode()).hexdigest()
            target=CACHE/key;temp=target.with_suffix(f'.{os.getpid()}-{threading.get_ident()}.tmp')
            temp.write_bytes(data[start-group[0]:start-group[0]+length]);temp.replace(target)
    with ThreadPoolExecutor(workers) as pool:
        for i,_ in enumerate(pool.map(read,groups),1):
            if i%10==0 or i==len(groups):print('Prefetch ranges',i,'/',len(groups),flush=True)


def remote_size(url):
    with tempfile.TemporaryDirectory(dir=CACHE) as directory:
        headers = Path(directory)/'headers'
        command = ['curl', '-4', '-L', '--fail', '-sS', '--max-time', '60',
                   '--max-filesize', '1048576', '--range', '0-0', '-D', str(headers),
                   '-o', os.devnull]
        result = run_curl(command, url + ('&' if '?' in url else '?') + 'sample_size_probe=1')
        if result.returncode:
            raise IOError(result.stderr.decode()[-300:])
        matches = re.findall(r'content-range:\s*bytes \d+-\d+/(\d+)', headers.read_text(), re.I)
        if not matches:
            raise IOError('HTTP range / size discovery not supported')
        return int(matches[-1])


class RemoteFile(io.RawIOBase):
    def __init__(self, url, size=None, block_size=256*1024):
        self.url, self.size = url, size if size is not None else remote_size(url)
        self.pos, self.block_size = 0, block_size

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos
    def read(self, n=-1):
        n = self.size-self.pos if n < 0 else min(n, self.size-self.pos)
        pieces = []
        while n > 0:
            start = self.pos // self.block_size * self.block_size
            length = min(self.block_size, self.size-start)
            chunk = fetch(self.url, start, length, max_bytes=max(length+1024, 1048576))
            offset = self.pos-start
            part = chunk[offset:offset+n]
            pieces.append(part); self.pos += len(part); n -= len(part)
        return b''.join(pieces)
    def readinto(self, buffer):
        data = self.read(len(buffer)); buffer[:len(data)] = data; return len(data)


class RemoteZip:
    """ZIP/ZIP64 central directory reader, also supporting split ZIP archives."""
    def __init__(self, urls, sizes=None):
        self.urls = [urls] if isinstance(urls, str) else urls
        self.sizes = sizes or [remote_size(u) for u in self.urls]
        self.starts = [0]
        for n in self.sizes: self.starts.append(self.starts[-1]+n)
        self.entries = self._index()
        self.by_name = {e['path']: e for e in self.entries}

    def span(self, start, length):
        result = []
        while length:
            disk = bisect.bisect_right(self.starts, start)-1
            count = min(length, self.starts[disk+1]-start)
            result.append(fetch(self.urls[disk], start-self.starts[disk], count,
                                max_bytes=max(count+1024, 1048576)))
            start += count; length -= count
        return b''.join(result)

    def _index(self):
        tail_length = min(65557, self.sizes[-1])
        tail = self.span(self.starts[-1]-tail_length, tail_length)
        end = tail.rfind(b'PK\x05\x06')
        if end < 0: raise ValueError('ZIP end record missing')
        eocd = struct.unpack_from('<4s4H2LH', tail, end)
        _, disk, cd_disk, n_disk, n_all, cd_size, cd_offset, comment = eocd
        if cd_size == 0xffffffff or cd_offset == 0xffffffff or n_all == 65535:
            locpos = end-20
            if tail[locpos:locpos+4] != b'PK\x06\x07': raise ValueError('ZIP64 locator missing')
            _, disk64, offset64, disks = struct.unpack_from('<4sLQL', tail, locpos)
            raw = self.span(self.starts[disk64]+offset64, 56)
            x = struct.unpack('<4sQ2H2L4Q', raw)
            cd_disk, n_all, cd_size, cd_offset = x[5], x[7], x[8], x[9]
        central = self.span(self.starts[cd_disk]+cd_offset, cd_size)
        records, pos = [], 0
        while pos < len(central):
            x = struct.unpack_from('<4s6H3L5H2L', central, pos)
            if x[0] != b'PK\x01\x02': raise ValueError('Bad central directory')
            flags, method, crc, compressed, size = x[3], x[4], x[7], x[8], x[9]
            namelen, extralen, commentlen, disk_start, offset = x[10], x[11], x[12], x[13], x[16]
            rawname = central[pos+46:pos+46+namelen]
            name = rawname.decode('utf-8' if flags & 2048 else 'cp437')
            extra = central[pos+46+namelen:pos+46+namelen+extralen]
            q = 0
            while q+4 <= len(extra):
                tag, count = struct.unpack_from('<HH', extra, q); data = extra[q+4:q+4+count]; q += 4+count
                if tag == 1:
                    k = 0
                    values = [size, compressed, offset, disk_start]
                    for i, sentinel in enumerate([0xffffffff]*3+[0xffff]):
                        if values[i] == sentinel:
                            fmt = '<L' if i == 3 else '<Q'; values[i] = struct.unpack_from(fmt, data, k)[0]; k += struct.calcsize(fmt)
                    size, compressed, offset, disk_start = values
            records.append(dict(path=name, size=size, compressed=compressed, method=method,
                                crc=crc, flags=flags, offset=self.starts[disk_start]+offset))
            pos += 46+namelen+extralen+commentlen
        if len(records) != n_all: raise ValueError((len(records), n_all))
        return records

    def read(self, name):
        e = self.by_name[name]
        if e['flags'] & 1: raise ValueError('Encrypted ZIP entry')
        count = min(e['compressed']+30+len(name.encode())+1024, self.starts[-1]-e['offset'])
        raw = self.span(e['offset'], count)
        h = struct.unpack_from('<4s5H3L2H', raw)
        if h[0] != b'PK\x03\x04': raise ValueError('Bad local header')
        begin = 30+h[-2]+h[-1]
        if begin+e['compressed'] > len(raw): raw = self.span(e['offset'], begin+e['compressed'])
        data = raw[begin:begin+e['compressed']]
        if e['method'] == 8: data = zlib.decompress(data, -15)
        elif e['method'] != 0: raise ValueError(f"Compression {e['method']} unsupported")
        if len(data) != e['size'] or zlib.crc32(data)&0xffffffff != e['crc']:
            raise ValueError(f'ZIP size/CRC mismatch: {name}')
        return data

    def extract(self, name, root):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe ZIP path')
        target = Path(root)/relative
        if target.exists() and target.stat().st_size == self.by_name[name]['size']:
            return target
        data = self.read(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix+'.part'); temp.write_bytes(data); temp.replace(target)
        return target


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)
