"""python -m hinglishguard.fetch_model  -  downloads the transformer weights if this copy of the project does not include them.

The small models (TF-IDF 'lite', language tagger, tokenizer) ship with the project; the transformer weights are
large, so a git clone fetches them from the release below. If the download is unavailable the app simply runs
the lite model, and you can retrain the transformer yourself with  run.bat pipeline.
"""
import hashlib
import sys
import urllib.request

from .paths import BUNDLE

# filled in when the weights are published as a release asset
WEIGHTS_URL = "https://huggingface.co/Vishnu-patil26/hinglishguard/resolve/main/weights.pt"
WEIGHTS_SHA256 = "dd0115653fc786833555ce27d7862c95684a19cb0965dbe70327f0f4d3a98a36"


def main() -> int:
    dest = BUNDLE / "weights.pt"
    if dest.exists():
        print("Transformer weights already present.")
        return 0
    if not WEIGHTS_URL:
        print("No download address is configured for the transformer weights - the app will use the lite model.")
        print("(Run `run.bat pipeline` to train the transformer yourself.)")
        return 0
    print(f"Downloading the transformer weights from {WEIGHTS_URL} ...")
    tmp = dest.with_suffix(".part")
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(WEIGHTS_URL, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
                h.update(block)
                done += len(block)
                if total:
                    print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
        print()
        if WEIGHTS_SHA256 and h.hexdigest() != WEIGHTS_SHA256:
            tmp.unlink(missing_ok=True)
            print("The downloaded file failed its checksum and was discarded - the app will use the lite model.")
            return 0
        tmp.replace(dest)
        print("Done.")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"Download failed ({exc}) - the app will use the lite model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
