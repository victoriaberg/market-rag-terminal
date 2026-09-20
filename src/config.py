from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_base_url: str = "https://models.github.ai/inference"
    llm_api_key: str = "" # Defaults to empty string if not provided in .env
    llm_model: str = "gpt-4o-mini"
    chroma_db_dir: str = "./chroma_db"
    default_tickers: list[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

    class Config:
        env_file = ".env"
        extra = "ignore" # Ignores any extra variables in the .env file

settings = Settings()