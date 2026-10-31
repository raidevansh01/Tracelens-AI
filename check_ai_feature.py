r"""Run the existing TraceLens experimental writing classifier on two QA samples.

Copy this file and its two TEST_*.txt companions into the TraceLens-AI project root.
Run: .\.venv\Scripts\python.exe check_ai_feature.py
Does not change or retrain the model.
"""
from pathlib import Path
from core.ai_patterns import infer, MODEL_FILE
from core.extractor import word_tokens

ROOT = Path(__file__).resolve().parent

print("Tracelens writing-pattern smoke test")
print("Model:", MODEL_FILE)
print("Model exists:", MODEL_FILE.is_file())

for name in ("TEST_AI_GENERATED.txt", "TEST_TOO_SHORT.txt"):
    path = ROOT / name
    if not path.is_file():
        print("MISSING TEST FILE:", path)
        continue
    content = path.read_text(encoding="utf-8")
    result = infer(content)
    print("\n===", name, "===")
    print("Words:", len(word_tokens(content)))
    print("Status:", result.get("status"))
    if result.get("status") == "experimental":
        print("Learned review threshold:", result["threshold"])
        windows = result.get("windows", [])
        print("Windows analyzed:", len(windows))
        for index, window in enumerate(windows, 1):
            print(
                f"  Window {index}: score={window['score']:.4f}, "
                f"flagged={window['review_flag']}, "
                f"characters={window['start']}:{window['end']}"
            )
        print("Evaluation reported by trained model:", result.get("evaluation"))
        print("NOTE: A known AI-written sample is not guaranteed to be flagged.")
        print("      A classifier score does not establish the author of a passage.")
    else:
        print("Details:", result.get("message", "No additional details"))

print("\nExpected: a trained model returns 'experimental' for the long file,")
print("and 'insufficient_text' for the short file. A review flag is NOT guaranteed.")
