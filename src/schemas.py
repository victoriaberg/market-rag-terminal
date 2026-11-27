from datetime import datetime
from pydantic import BaseModel

class NewsArticle(BaseModel):
    ticker: str
    title: str
    publisher: str
    link: str
    publish_time: datetime
    summary: str

class StockSnapshot(BaseModel):
    ticker: str
    current_price: float
    day_high: float
    day_low: float
    volume: int
    timestamp: datetime

class RetrievedArticle(BaseModel):
    article_id: str
    ticker: str
    title: str
    publisher: str
    link: str
    publish_time: datetime
    document: str
    semantic_score: float
    recency_score: float
    final_score: float