from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    reasoning_api_base: str = Field(min_length=1)
    reasoning_api_key: SecretStr = Field(min_length=1)
    log_level: str = "INFO"
    reasoning_model: str = "glm-5.3-flash"
    reasoning_effort: str = "low"
    reasoning_max_tokens: int = 400
    history_turns: int = Field(default=10, ge=0)
    scene_model: str = ""
    scene_effort: str = "high"
    scene_max_tokens: int = Field(default=128000, gt=0)
    scene_timeout_s: float = Field(default=600.0, gt=0)
    planner_model: str = ""
    planner_effort: str = "high"
    planner_max_tokens: int = Field(default=64000, gt=0)
    planner_timeout_s: float = Field(default=120.0, gt=0)
    signaling_port: int = 8080


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
