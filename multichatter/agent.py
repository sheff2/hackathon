import os
from google.adk.agents import Agent, SequentialAgent

# Tools
from multichatter.utils.agent1_utils import plan_routes
from multichatter.utils.agent2_utils import (
    rank_supplied_routes,
    get_route_details,
)

# Env-driven model selection
ROUTER_MODEL = os.getenv("GOOGLE_GENLANG_MODEL_ROUTER", os.getenv("GOOGLE_GENLANG_MODEL", "gemini-2.5-flash"))
CRIME_MODEL  = os.getenv("GOOGLE_GENLANG_MODEL_CRIME",  os.getenv("GOOGLE_GENLANG_MODEL", "gemini-2.5-flash"))

# -------- Agent 1: routing (SILENT) --------
router_agent = Agent(
    name="routing_agent",
    model=ROUTER_MODEL,
    description="Finds up to 10 distinct walking routes (machine output only).",
    instruction=(
        "You MUST call plan_routes(origin, destination, count=10, output='json'). "
        "After the tool returns: DO NOT send any message to the USER. "
        "Return no user-facing text; the next agent will speak to the user. "
        "The tool result (status/origin/destination/routes) will be available to the next agent."
    ),
    tools=[plan_routes],
)

# -------- Agent 2: crime scoring + summary (user-facing) --------
crime_agent = Agent(
    name="crime_route_agent",
    model=CRIME_MODEL,
    description="Scores walking routes using local incident data, then summarizes clearly for the user.",
    instruction=(
        "Find the most recent function/tool RESPONSE from routing_agent with keys: status, origin, destination, routes. "
        "If missing OR status != 'success': apologize briefly and ask the user to restate origin and destination with city/state. Stop.\n"
        "Otherwise, call rank_supplied_routes with EXACT JSON using those values:\n"
        '{\"routes\": RESPONSE.routes, \"origin\": RESPONSE.origin, \"destination\": RESPONSE.destination}\n'
        "After the tool returns, speak to the USER. Start with: "
        "'Here you go — I generated N routes and ranked their safety.' (replace N). "
        "Then show the TOP 3 with: rank, duration (min), distance (km), risk_summary, maps_link. "
        "Be concise and never paste raw polylines.\n"
        "If the USER asks about a specific route_id, call get_route_details with "
        '{\"route_id\": \"<the_id>\"} and briefly explain key risk factors.'
    ),
    tools=[rank_supplied_routes, get_route_details],
)

# -------- Pipeline --------
code_pipeline_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[router_agent, crime_agent],
    description="1) Get routes as JSON (machine output). 2) Rank those routes and produce a user-facing safety summary.",
)

# ADK entrypoint
root_agent = code_pipeline_agent