"""Convenience wrapper around the OpenAI Responses API for GPT-5 models."""

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class GPT5Client:
    """Provide a thin client facade for invoking GPT-5 style Responses APIs."""

    def __init__(self, api_key=None, model=None):
        """Initialize a reusable OpenAI client.

        Parameters
        ----------
        api_key : str | None
            Optional explicit API key value; defaults to the environment variable used by the SDK.
        model : str | None
            Optional model identifier; falls back to ``LLM_MODEL`` environment variable or ``"gpt-5"``.
        """

        self.model = model or os.getenv("LLM_MODEL", "gpt-5")
        # The official SDK pulls API key from env if not provided
        self.client = OpenAI(api_key=api_key)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        reasoning: str = "high",           # "low" | "medium" | "high"
        max_output_tokens: int = 4000,
        text: dict | None = None,          # structured outputs live here
    ):
        """Execute a single-turn LLM request with optional structured output.

        Parameters
        ----------
        system_prompt : str
            Instructional text supplied to the model as the system role content.
        user_prompt : str
            The concrete user query or payload to be answered by the model.
        reasoning : str, default "high"
            Reasoning effort hint for the API; typically ``"low"``, ``"medium"``, or ``"high"``.
        max_output_tokens : int, default 4000
            Upper bound on the number of tokens generated in the response.
        text : dict | None
            Optional structured output configuration passed directly to the API.

        Returns
        -------
        str | dict
            Parsed structured result when available, otherwise the raw response text.
        """
        resp = self.client.responses.create(
            model=self.model,
            reasoning={"effort": reasoning},
            max_output_tokens=max_output_tokens,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user",   "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            **({"text": text} if text is not None else {}),
        )

        # If using structured outputs ("json" or "json_schema"), SDK often provides parsed form.
        # Fallback to raw text when parsed isn't present.
        return getattr(resp, "output_parsed", None) or resp.output_text
