"""Give common.openrouter (and so judges.*) a connection pool large enough for thousands of calls in flight.
common's default (48 connections) would serialize the build's judge calls. Import before the first call."""
import httpx
from openai import OpenAI

import common

common._OR["c"] = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=common.os.environ["OPENROUTER_API_KEY"], max_retries=0,
    timeout=1200, http_client=httpx.Client(limits=httpx.Limits(max_connections=6000, max_keepalive_connections=2000),
                                           timeout=httpx.Timeout(1200.0, connect=30.0)))
