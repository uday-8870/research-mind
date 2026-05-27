"""
Tests for ResearchMind agents.
Run: pytest tests/ -v
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from backend.utils.chunker import chunk_text, count_tokens
from backend.agents.reader_agent import clean_html, extract_content


# ── Chunker Tests ─────────────────────────────────────────────────
class TestChunker:
    def test_basic_chunking(self):
        text = " ".join(["This is a sentence."] * 100)
        chunks = chunk_text(text, max_tokens=100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= 120  # Allow slight overflow on sentence boundary

    def test_overlap(self):
        text = " ".join(["Sentence number {}.".format(i) for i in range(50)])
        chunks = chunk_text(text, max_tokens=80, overlap_tokens=20)
        # Check that adjacent chunks share some content
        if len(chunks) >= 2:
            assert len(chunks[0].text) > 0
            assert len(chunks[1].text) > 0

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_short_text_single_chunk(self):
        text = "This is a short text."
        chunks = chunk_text(text)
        assert len(chunks) <= 1


# ── Reader Tests ──────────────────────────────────────────────────
class TestReaderAgent:
    def test_clean_html_strips_tags(self):
        html = "<html><body><script>alert(1)</script><p>Hello world</p></body></html>"
        result = clean_html(html)
        assert "Hello world" in result
        assert "<script>" not in result
        assert "<p>" not in result

    def test_clean_html_empty(self):
        assert clean_html("") == ""

    def test_extract_content_prefers_raw(self):
        source = {
            "raw_content": "<p>" + "A" * 600 + "</p>",
            "content": "Short snippet",
        }
        result = extract_content(source)
        assert len(result) > 100

    def test_extract_content_falls_back_to_snippet(self):
        source = {"raw_content": "", "content": "Just a snippet"}
        result = extract_content(source)
        assert result == "Just a snippet"


# ── Eval Tests ────────────────────────────────────────────────────
class TestMetrics:
    @pytest.mark.asyncio
    async def test_evaluate_report_returns_result(self):
        """Integration test — mocks the LLM calls."""
        mock_response = MagicMock()
        mock_response.content = '{"faithfulness_score": 0.85, "unsupported_claims": [], "reasoning": "good"}'

        with patch("backend.utils.metrics.get_eval_llm") as mock_llm:
            mock_instance = AsyncMock()
            mock_instance.ainvoke.return_value = mock_response
            mock_llm.return_value = mock_instance

            from backend.utils.metrics import evaluate_report
            result = await evaluate_report(
                query="What is quantum computing?",
                sub_queries=["How does it work?", "What are the applications?"],
                report="Quantum computing uses qubits. It has many applications.",
                sources=[{"url": "https://example.com", "content": "Quantum computing explanation"}],
            )

        assert 0 <= result.faithfulness <= 1
        assert 0 <= result.coverage <= 1
        assert isinstance(result.flagged_claims, list)


# ── Orchestrator Integration Test ─────────────────────────────────
class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_run_research_structure(self):
        """Test that orchestrator returns expected state structure."""
        with patch("backend.agents.orchestrator.run_search_agent") as mock_search, \
             patch("backend.agents.orchestrator.run_reader_agent") as mock_reader, \
             patch("backend.agents.orchestrator.run_synthesis_agent") as mock_synth, \
             patch("backend.agents.orchestrator.run_critic_agent") as mock_critic:

            mock_search.return_value = (["sub q1", "sub q2"], [{"url": "http://a.com", "content": "text", "title": "A", "score": 0.9}])
            mock_reader.return_value = [{"url": "http://a.com", "content": "text", "title": "A", "score": 0.9, "chunk_count": 5}]
            mock_synth.return_value = "# Research Report\n\nFindings..."
            mock_critic.return_value = None

            from backend.agents.orchestrator import run_research
            result = await run_research("What is machine learning?", depth="quick")

        assert "report" in result
        assert "sub_queries" in result
        assert result["status"] == "done"
        assert result["report"] == "# Research Report\n\nFindings..."
