#!/usr/bin/env python3
"""Read a possibly TRUNCATED (still downloading or stopped) zip of stored entries by walking its local file headers, so scans
that are completely present can be used without waiting for the central directory at the end of the file.
  zip_stream.py list <zip>                    print names of complete .h5 entries
  zip_stream.py get  <zip> <name> <out>       copy one entry to <out>"""
import sys, os, struct
def entries(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        pos = 0
        while pos + 30 <= size:
            f.seek(pos); h = f.read(30)
            if h[:4] != b"PK\x03\x04": return
            _, flags, method, _, _, _, csize, usize, nlen, xlen = struct.unpack("<HHHHHIIIHH", h[4:])
            name = f.read(nlen).decode(); data = pos + 30 + nlen + xlen
            if method != 0 or (flags & 8): raise SystemExit("entry %s is compressed or uses a data descriptor" % name)
            if data + csize > size: return                                   # incomplete entry: stop
            yield name, data, csize
            pos = data + csize
if __name__ == "__main__":
    if sys.argv[1] == "list":
        for n, _, _ in entries(sys.argv[2]):
            if n.endswith(".h5"): print(n)
    else:
        for n, off, sz in entries(sys.argv[2]):
            if n == sys.argv[3]:
                with open(sys.argv[2], "rb") as f, open(sys.argv[4], "wb") as o:
                    f.seek(off); left = sz
                    while left: b = f.read(min(left, 1 << 22)); o.write(b); left -= len(b)
                sys.exit(0)
        raise SystemExit("not found")
