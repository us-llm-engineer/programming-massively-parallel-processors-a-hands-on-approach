#!/usr/bin/env python3
"""Fetch ONE real multi-coil brain k-space file from the M4Raw dataset (Zenodo record 8056074, CC-BY 4.0) by reading the zip's
central directory with HTTP range requests, so the 3 GB archive is not downloaded. Usage: fetch_m4raw.py [list|get NAME] [dest]"""
import sys, io, zipfile, requests
URL = "https://zenodo.org/api/records/8056074/files/M4RawV1.5_multicoil_val.zip/content"
class Ranged(io.RawIOBase):
    def __init__(s): s.pos = 0; s.size = int(requests.head(URL, allow_redirects=True).headers["Content-Length"])
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, o, w=0): s.pos = o if w == 0 else s.pos + o if w == 1 else s.size + o; return s.pos
    def read(s, n=-1):
        if n < 0 or s.pos + n > s.size: n = s.size - s.pos
        if n <= 0: return b""
        r = requests.get(URL, headers={"Range": f"bytes={s.pos}-{s.pos+n-1}"}); r.raise_for_status(); s.pos += len(r.content); return r.content
    def readinto(s, b): d = s.read(len(b)); b[:len(d)] = d; return len(d)
z = zipfile.ZipFile(io.BufferedReader(Ranged(), 1 << 20))
if sys.argv[1] == "list":
    for i in z.infolist()[:40]: print(i.filename, i.file_size, i.compress_size)
    print(len(z.infolist()), "files")
else:
    dest = sys.argv[3] if len(sys.argv) > 3 else "data/m4raw"; z.extract(sys.argv[2], dest); print("extracted", sys.argv[2], "to", dest)
