# hackathon

emergency contact agent

real time check in agent python/agents/realtime-conversational-agent

python/agents/image-scoring → Street View brightness scoring
Borrow its “ingest image → score” pattern; point it at Street View Static tiles per route segment to estimate luminance / lamppost presence. (There’s active discussion around the image tool usage in that sample, so it’s live and relevant.

python/agents/safety-plugins → safety guardrails
Use plugins to gate tool I/O and panic phrases (e.g., long-press SOS) before sending buddy alerts. The ADK safety docs show the plugin approach for guardrails

python/agents/RAG → local open-data (sidewalks/ramps/lights)
crime data etc
