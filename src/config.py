from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "https://models.github.ai/inference"
    llm_api_key: str = ""  # Defaults to empty string if not provided in .env
    llm_model: str = "gpt-4o-mini"
    chroma_db_dir: str = "./chroma_db"
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


settings = Settings()