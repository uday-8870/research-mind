"""
Smart Text Chunker — respects sentence boundaries and semantic structure.
Uses tiktoken for accurate token counting.
"""
import re
import tiktoken
from dataclasses import dataclass

ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    token_count: int
    char_start: int
    char_end: int


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def chunk_text(
    text: str,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    min_chunk_tokens: int = 50,
) -> list[Chunk]:
    """
    Split text into overlapping chunks that respect sentence boundaries.
    Much better than naive character splitting.
    """
    # Clean whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    # Split into sentences (handles abbreviations reasonably)
    sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    sentences = sentence_pattern.split(text)

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    char_pos = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # If single sentence exceeds max, split by words
        if sentence_tokens > max_tokens:
            if current_sentences:
                _flush_chunk(chunks, current_sentences, char_pos)
                char_pos += sum(len(s) + 1 for s in current_sentences)
                current_sentences, current_tokens = [], 0

            words = sentence.split()
            word_buffer: list[str] = []
            word_tokens = 0
            for word in words:
                wt = count_tokens(word + " ")
                if word_tokens + wt > max_tokens and word_buffer:
                    chunk_text_str = " ".join(word_buffer)
                    chunks.append(
                        Chunk(
                            text=chunk_text_str,
                            token_count=word_tokens,
                            char_start=char_pos,
                            char_end=char_pos + len(chunk_text_str),
                        )
                    )
                    char_pos += len(chunk_text_str) + 1
                    word_buffer = word_buffer[-5:]  # overlap
                    word_tokens = count_tokens(" ".join(word_buffer))
                word_buffer.append(word)
                word_tokens += wt
            if word_buffer:
                chunk_text_str = " ".join(word_buffer)
                chunks.append(
                    Chunk(
                        text=chunk_text_str,
                        token_count=word_tokens,
                        char_start=char_pos,
                        char_end=char_pos + len(chunk_text_str),
                    )
                )
            continue

        if current_tokens + sentence_tokens > max_tokens and current_sentences:
            _flush_chunk(chunks, current_sentences, char_pos)
            char_pos += sum(len(s) + 1 for s in current_sentences)

            # Keep overlap sentences
            overlap_buffer: list[str] = []
            overlap_count = 0
            for s in reversed(current_sentences):
                st = count_tokens(s)
                if overlap_count + st <= overlap_tokens:
                    overlap_buffer.insert(0, s)
                    overlap_count += st
                else:
                    break
            current_sentences = overlap_buffer
            current_tokens = overlap_count

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    if current_sentences:
        _flush_chunk(chunks, current_sentences, char_pos)

    # Filter tiny chunks
    return [c for c in chunks if c.token_count >= min_chunk_tokens]


def _flush_chunk(chunks: list[Chunk], sentences: list[str], char_pos: int):
    text = " ".join(sentences)
    chunks.append(
        Chunk(
            text=text,
            token_count=count_tokens(text),
            char_start=char_pos,
            char_end=char_pos + len(text),
        )
    )
