from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # LLM and embedding settings
    llm_base_url: str = "https://models.github.ai/inference"
    llm_api_key: str = ""  # Defaults to empty string if not provided in .env
    llm_model: str = "gpt-4o-mini"
    chroma_db_dir: str = "./chroma_db"
    chroma_collection_name: str = "market_news"
    embedding_model: str = "all-MiniLM-L6-v2"
    default_tickers: list[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    # Company names used to check that a news article is really about a ticker.
    ticker_aliases: dict[str, list[str]] = {
        "AAPL": ["Apple"],
        "MSFT": ["Microsoft"],
        "NVDA": ["Nvidia"],
        "GOOGL": ["Google", "Alphabet"],
        "AMZN": ["Amazon"],
    }
    poll_interval_seconds: int = 180
    news_per_ticker: int = 10
    # Scoring knobs for ranking articles
    decay_half_life_hours: float = 24.0
    recency_weight: float = 0.3

# Create a single global settings instance that can be imported and used throughout the application.
settings = Settings()