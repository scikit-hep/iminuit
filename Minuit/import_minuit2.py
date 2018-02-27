import os
import re
import argparse
import shutil
from glob import glob
from collections import Counter

def scan(d, src_file):
    s = open(src_file).read()
    matches = re.findall(r"^\s*#\s*include\s*\"([^\"]+)\.h\"", s, re.MULTILINE)
    for m in matches:
        key = os.path.basename(m)
        if key not in d:
            d[key] = None

def mirror(prefixes, src, dst):
    for prefix in prefixes:
        for t in ("inc", "dst"):
            if src.startswith(prefix + "/" + t):
                s = src[len(prefix)+5:]
                d = os.path.dirname(s)
                if d:
                    dst = dst + "/" + d
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                shutil.copy(src, dst)
                return

class FilePair(object):
    def __init__(self):
        self.inc = None
        self.src = None
    def __call__(self, x):
        if x.endswith(".h"):
            self.inc = x
        else:
            self.src = x
    def __str__(self):
        if self.src is None:
            return self.inc
        return "%s, %s" % (self.inc, self.src)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dirs", nargs="+",
                        help="Source directories with Minuit2 files in ROOT")

    args = parser.parse_args()

    imported = dict()
    for pyx_file in glob("iminuit/*.pxi") + glob("iminuit/*.pyx"):
        imported.update({x:None for x in re.findall(r"extern from \"Minuit2/([^\"]+).h\"", open(pyx_file).read())})

    src_files = dict()
    for sdir in args.source_dirs:
        for sf in glob(sdir + "/inc/*.h") + glob(sdir + "/inc/*/*.h") + glob(sdir + "/src/*.*"):
            base = os.path.splitext(os.path.basename(sf))[0]
            if base not in src_files:
                src_files[base] = FilePair()
            src_files[base](sf)

    while Counter(imported.values())[None] > 0:
        for key, value in imported.items():
            if value is None:
                if key not in src_files:
                    imported[key] = "NOT-FOUND"
                else:
                    fp = src_files[key]
                    imported[key] = fp
                    if fp.inc is not None:
                        scan(imported, fp.inc)
                    if fp.src is not None:
                        scan(imported, fp.src)

    for k,v in imported.items():
        if v == "NOT-FOUND":
            print "Missing:",k
            continue
        mirror(args.source_dirs, v.inc, "Minuit/inc")
        if v.src is not None:
            shutil.copy(v.src, "Minuit/src")

if __name__ == '__main__':
    main()