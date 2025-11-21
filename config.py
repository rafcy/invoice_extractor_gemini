"""
Configuration management for the Invoice Processing API
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Google Gemini API
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")

    # Rate Limiting
    rate_limit_max_per_minute: int = Field(default=10, alias="RATE_LIMIT_MAX_PER_MINUTE")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")

    # Caching
    enable_cache: bool = Field(default=True, alias="ENABLE_CACHE")
    max_cache_size: int = Field(default=100, alias="MAX_CACHE_SIZE")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings singleton"""
    return Settings()
