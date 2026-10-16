"""python -m hinglishguard.selfcheck  -  proves the installed model loads and answers (used by run.bat / run.sh)."""
import sys

from . import inference as I


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    det = I.load()
    r = det.analyse(["Yaar yeh movie toh ekdum paisa vasool thi, maza aa gaya"])[0]
    ok = all(abs(sum(p.values()) - 1.0) < 1e-3 for p in r.probs.values()) and 0.0 <= r.toxicity <= 1.0
    print(f"Model: {det.backend}.  Test sentence -> sentiment {r.label('sentiment')}, toxicity {r.toxicity:.2f} ({r.action}).")
    print("Setup check passed." if ok else "Setup check FAILED: the model returned invalid probabilities.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
