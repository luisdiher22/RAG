#This is a model for structuring search queries with metadata filters for tutorial videos about a software library
import datetime
from typing import Literal, Optional, Tuple
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

class TutorialSearch(BaseModel):
    """ Search over a db of tutorial videos about a software library"""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts",
    )

    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles." \
            "Should be succint and only contain keywords that could be in a title."
        ),
    )

    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive. Only use if explicitly specified.",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use if explicitly specified.",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
    )

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")