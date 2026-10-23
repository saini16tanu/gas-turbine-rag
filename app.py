"""CLI.  python app.py "your question"  -> answer one question
         python app.py                  -> run the 5 test questions"""
import sys
import textwrap

from rag import ask

TEST_QUESTIONS = [
    "What are the three main types of inspection?",
    "What does a maintenance factor of two mean?",
    "What is the purchase price of this gas turbine?",
    "What is the melting point of titanium?",
    "What is the recommended combustion-inspection interval in hours?",
]


def show(label: str, question: str) -> None:
    try:
        result = ask(question)
    except Exception as err:  # e.g. missing API key, quota or network error
        print(f"{label}Question : {question}\nError    : {type(err).__name__}: {str(err)[:200]}\n")
        return
    indent = " " * 11
    wrap = lambda text: "\n".join(  # noqa: E731 - wrap each line, keep bullets/newlines
        textwrap.fill(line, 100, initial_indent=indent, subsequent_indent=indent)
        for line in text.splitlines() if line.strip()
    ).lstrip()
    print("=" * 100)
    print(f"{label}Question : {question}")
    print("-" * 100)
    print(f"Answer   : {wrap(result['answer'])}")
    print(f"Reference: {wrap(result['reference'] or '-')}")
    if result["evidence"]:  # no evidence for "Not found"
        quoted = f'"{result["evidence"]}" (page {result["evidence_page"]})'
        print(f"Evidence : {wrap(quoted)}")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show("", " ".join(sys.argv[1:]))
    else:
        for i, q in enumerate(TEST_QUESTIONS, 1):
            show(f"Q{i} ", q)
