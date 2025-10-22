#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Dict, Any


def read_text_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Dict[str, Any]]:
    """Chunk text using LangChain's RecursiveCharacterTextSplitter if available,
    falling back to a simple paragraph+sliding-window splitter.
    Returns a list of dicts: {text, start, end}.
    """
    splitter = None
    try:
        # Prefer the new package layout
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        splits = splitter.split_text(text)
        # Approximate character spans by cumulative lengths
        chunks: List[Dict[str, Any]] = []
        cursor = 0
        for s in splits:
            start = text.find(s, cursor)
            if start == -1:
                start = cursor
            end = start + len(s)
            chunks.append({"text": s, "start": start, "end": end})
            cursor = end
        return chunks
    except Exception:
        try:
            # Older import path
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            splits = splitter.split_text(text)
            chunks: List[Dict[str, Any]] = []
            cursor = 0
            for s in splits:
                start = text.find(s, cursor)
                if start == -1:
                    start = cursor
                end = start + len(s)
                chunks.append({"text": s, "start": start, "end": end})
                cursor = end
            return chunks
        except Exception:
            pass

    # Fallback splitter: paragraph merge with sliding window on words
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunks: List[Dict[str, Any]] = []
    current: List[str] = []
    current_len = 0
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        para_len = len(para)
        if current_len + para_len + (2 if current else 0) <= chunk_size:
            current.append(para)
            current_len += para_len + (2 if current_len > 0 else 0)
            i += 1
        else:
            if not current:
                # Single very long paragraph: split by words
                words = para.split()
                start_idx = 0
                while start_idx < len(words):
                    end_idx = min(start_idx + chunk_size, len(words))
                    piece = " ".join(words[start_idx:end_idx])
                    chunks.append(_locate_span_in_text(text, piece))
                    if end_idx >= len(words):
                        break
                    start_idx = max(end_idx - chunk_overlap, 0)
                i += 1
            else:
                piece = "\n\n".join(current)
                chunks.append(_locate_span_in_text(text, piece))
                # For overlap: move back by overlap size in characters heuristically
                if chunk_overlap > 0:
                    back_chars = min(chunk_overlap, len(piece))
                    back_text = piece[-back_chars:]
                    # Re-initialize with overlap tail if useful
                    current = [back_text]
                    current_len = len(back_text)
                else:
                    current = []
                    current_len = 0
    if current:
        piece = "\n\n".join(current)
        chunks.append(_locate_span_in_text(text, piece))
    return chunks


def _locate_span_in_text(full_text: str, span_text: str) -> Dict[str, Any]:
    start = full_text.find(span_text)
    if start == -1:
        # Best effort: approximate by length
        return {"text": span_text, "start": 0, "end": len(span_text)}
    end = start + len(span_text)
    return {"text": span_text, "start": start, "end": end}


def load_embedder(model_name: str):
    # Prefer new provider packages, then fall back to legacy import path
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore

            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception as exc:
            raise RuntimeError(
                "Could not import HuggingFaceEmbeddings from LangChain. Please install 'langchain-huggingface' or 'langchain'."
            ) from exc


def embed_chunks(chunks: List[Dict[str, Any]], embedder) -> List[List[float]]:
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_documents(texts)
    return vectors


def write_jsonl(
    chunks: List[Dict[str, Any]],
    vectors: List[List[float]],
    output_path: str,
    source_path: str,
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("Chunks and vectors length mismatch")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            record = {
                "id": f"{os.path.basename(source_path)}::chunk_{i}",
                "text": chunk["text"],
                "embedding": vec,
                "metadata": {
                    "source": os.path.abspath(source_path),
                    "chunk_index": i,
                    "start_char": chunk.get("start", None),
                    "end_char": chunk.get("end", None),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk a text file and generate embeddings for RAG. Outputs JSONL.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="mulholland_veri.txt",
        help="Path to input text file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chunks.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size (characters/tokens heuristic)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="Chunk overlap (characters/tokens heuristic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name for embeddings",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    text = read_text_file(args.input)
    chunks = chunk_text(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    embedder = load_embedder(args.model)
    vectors = embed_chunks(chunks, embedder)
    write_jsonl(chunks, vectors, args.output, args.input)
    print(
        f"Wrote {len(chunks)} chunk embeddings to {os.path.abspath(args.output)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))



