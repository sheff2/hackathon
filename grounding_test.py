from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleMaps,
    HttpOptions,
    Tool,
)

client = genai.Client(http_options=HttpOptions(api_version="v1"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Where can I get the best espresso near me?",
    config=GenerateContentConfig(
        tools=[
            # Use Google Maps Tool
            Tool(google_maps=GoogleMaps())
        ],
        tool_config=types.ToolConfig(
            retrieval_config = types.RetrievalConfig(
                lat_lng = types.LatLng( # Pass coordinates for location-aware grounding
                    latitude=40.7128,
                    longitude=-74.006
                ),
                language_code = "en_US", # Optional: localize Maps results
            ),
        ),
    ),
)

print(response.text)
# Example response:
# 'Here are some of the top-rated places to get espresso near you: ...'