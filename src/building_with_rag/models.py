from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Pattern = Literal[
    "semantic", "hybrid", "hybrid-reranked", "structured", "decomposition", "hyde"
]


class SemanticFilters(BaseModel):
    act: list[str] = []
    status: list[str] = []
    access_level: list[str] = []


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    pattern: Pattern
    caller_id: str | None = None
    filters: SemanticFilters | None = None
    limit: int = Field(default=5, ge=1, le=20)
    generate_answer: bool = False
    required_acts: list[str] = []
    chapter: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str | None = None
    score: float | None = None
    source: dict[str, Any] = {}


class GenerationResult(BaseModel):
    outcome: Literal["answered", "insufficient_evidence", "unavailable", "malformed"]
    answer: str | None = None
    claims: list[Any] = []
    citations: list[Any] = []
    supporting_passages: list[Any] = []
    provider: str | None = None
    model: str | None = None
    trace: list[Any] = []
    context_outcome: str | None = None
    confidence: float | None = None
    draft_answer: str | None = None
    issues: list[Any] = []
    attempts: int = 0
    low_confidence_reason: str | None = None


class OmittedCandidate(BaseModel):
    chunk_id: str
    omitted_reason: str


class SubquestionEvidence(BaseModel):
    subquestion: str
    status: Literal["evidenced", "no_evidence"]
    results: list[RetrievedChunk] = []
    reason: str | None = None


class StructuredSignals(BaseModel):
    intent: Literal["exact_lookup", "filter", "aggregation"]
    act: str | None = None
    section_number: str | None = None
    chapter: str | None = None


class QueryResult(BaseModel):
    pattern: str
    status: str
    message: str
    trace: list[Any] = []
    results: list[RetrievedChunk] = []
    generation: GenerationResult | None = None
    omitted_candidates: list[OmittedCandidate] = []
    subquestions: list[SubquestionEvidence] = []
    hyde_direct_candidates: list[RetrievedChunk] = []
    hyde_query_candidates: list[RetrievedChunk] = []
    hyde_hypothetical_text_debug: str | None = None


# --- OpenAI contracts ---
class RagOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: Pattern | None = None
    act: list[str] = []
    status: list[str] = []
    access_level: list[str] = []
    limit: int | None = Field(default=None, ge=1, le=20)
    required_acts: list[str] = []
    chapter: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)
    rag_options: RagOptions | None = None


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class ChatDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatChunkChoice(BaseModel):
    index: int = 0
    delta: ChatDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "building-with-rag"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


# --- MongoDB-schema contracts (typed documents only) ---
class SectionDocument(BaseModel):
    chunk_id: str
    section_id: str
    act: str
    text: str
    heading: str | None = None
    status: str | None = None
    access_level: str | None = None
    chapter: str | None = None
    source: dict[str, Any] = {}


class ChunkEmbeddingDocument(SectionDocument):
    embedding: list[float] = []
    embedding_model: str = "voyage-3.5"
    embedding_version: str = "voyage-3.5"
    embedding_dimensions: int = 1024
