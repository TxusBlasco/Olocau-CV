import argparse
from rag_utils import build_chain
from config import TOP_K

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Query string")
    ap.add_argument("--k", type=int, default=TOP_K, help="Top-k results")
    args = ap.parse_args()

    chain = build_chain(top_k=args.k)
    res = chain.invoke({"query": args.q})

    print("---- ANSWER ----")
    print(res["result"])
    if res.get("source_documents"):
        print("\n---- SOURCES ----")
        for i, doc in enumerate(res["source_documents"], 1):
            print(f"[{i}] {doc.metadata.get('file')}")

if __name__ == "__main__":
    main()
