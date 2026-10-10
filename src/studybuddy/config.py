from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "gemini"  # gemini | groq | openrouter | ollama
    llm_model: str = "gemini-3.5-flash-lite"
    google_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    top_k: int = 4
    min_relevance: float = 0.5
    progress_db: str = ""
    embed_cache_db: str = "./.cache/embeddings.db"

    def llm_ready(self) -> bool:
        keys = {
            "gemini": self.google_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
            "ollama": "no key needed",
        }
        return bool(keys.get(self.llm_provider))


settings = Settings()