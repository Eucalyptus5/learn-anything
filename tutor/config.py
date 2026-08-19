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


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
