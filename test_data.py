"""python test_data.py -- the manifest check catches missing, resized and altered files."""
import hashlib
import os
import tempfile

from data import verify

with tempfile.TemporaryDirectory() as root:
    open(os.path.join(root, "a.txt"), "wb").write(b"hello")
    files = {"a.txt": {"bytes": 5, "sha256": hashlib.sha256(b"hello").hexdigest()},
             "b.txt": {"bytes": 1, "sha256": "0" * 64}}
    assert verify(root, files) == [("b.txt", "missing")]
    open(os.path.join(root, "a.txt"), "wb").write(b"hellO")  # same size, different content
    assert verify(root, files, quick=True) == [("b.txt", "missing")]
    assert verify(root, files)[0] == ("a.txt", "sha256 differs")
    open(os.path.join(root, "a.txt"), "wb").write(b"hi")
    assert verify(root, files)[0][1].startswith("size")
print("ok")
