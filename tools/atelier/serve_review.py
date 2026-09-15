#!/usr/bin/env python3
"""Serve the finished local art gallery with byte-range support for film seeking."""

import argparse
import functools
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO


class ReviewHandler(SimpleHTTPRequestHandler):
    """Add single byte ranges while keeping the gallery on the loopback interface."""

    byte_range: tuple[int, int] | None = None

    def send_head(self) -> BinaryIO | None:
        self.byte_range = None
        path = Path(self.translate_path(self.path))
        requested = self.headers.get("Range")
        if not requested or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
        size = path.stat().st_size
        if not match or not any(match.groups()) or size == 0:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
        a, b = match.groups()
        if a:
            first = int(a)
            last = min(int(b) if b else size - 1, size - 1)
        else:
            first, last = max(0, size - int(b)), size - 1
        if first >= size or first > last:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        try:
            stream = path.open("rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return None
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.send_header("Content-Length", str(last - first + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        stream.seek(first)
        self.byte_range = first, last
        return stream

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        try:
            if self.byte_range is None:
                super().copyfile(source, outputfile)
                return
            first, last = self.byte_range
            remaining = last - first + 1
            while remaining > 0:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    break
                outputfile.write(block)
                remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            # Seeking or replacing a film deliberately cancels an old transfer.
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    root = args.directory.resolve()
    if not (root / "index.html").is_file():
        parser.error("Build the review gallery in this directory first")
    handler = functools.partial(ReviewHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Review gallery: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
