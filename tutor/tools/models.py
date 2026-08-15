from pydantic import BaseModel, ConfigDict, Field

MAX_COLUMNS = 400

PATH_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".c",
        ".cc",
        ".cfg",
        ".cpp",
        ".cs",
        ".css",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".md",
        ".py",
        ".pyi",
        ".rb",
        ".rs",
        ".rst",
        ".sh",
        ".sql",
        ".swift",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)


class Position(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    line: int | None = None
    symbol: str | None = None

    def key(self) -> tuple[str, int | None]:
        return (self.path, self.line)


class ContextLine(BaseModel):
    line: int
    text: str


class SearchMatch(BaseModel):
    path: str
    line: int
    text: str
    before: list[ContextLine] = Field(default_factory=list)
    after: list[ContextLine] = Field(default_factory=list)


class SearchResult(BaseModel):
    tool: str
    query: str
    globs: list[str]
    matches: list[SearchMatch]
    truncated: bool
    oversized: bool
    byte_count: int

    def positions(self) -> list[Position]:
        positions: list[Position] = []
        for match in self.matches:
            for context_line in match.before:
                positions.append(Position(path=match.path, line=context_line.line))
            positions.append(Position(path=match.path, line=match.line))
            for context_line in match.after:
                positions.append(Position(path=match.path, line=context_line.line))
        return positions


class GroundingVerdict(BaseModel):
    ok: bool
    ungrounded: list[Position] = Field(default_factory=list)


class SearchBudget(BaseModel):
    max_matches: int = Field(default=40, gt=0)
    max_bytes: int = Field(default=24000, gt=0)
    max_record_bytes: int = Field(default=1000000, gt=0)
    context_lines: int = Field(default=2, ge=0)
    timeout_ms: int = Field(default=2000, gt=0)
