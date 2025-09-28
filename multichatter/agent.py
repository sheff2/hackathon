# agent.py
import sys
import os
from pathlib import Path

# Add the parent directory to Python path
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from google.adk.agents import Agent, SequentialAgent

# Now this will always work
from multichatter.utils.agent1_utils import plan_routes
from multichatter.utils.agent2_utils import (
    rank_supplied_routes,
    parse_and_rank_routes,
    get_route_details,
    debug_agent_context,
    get_crime_locations,
    get_ranked_routes,
)

# -------- Agent 1: Routing Agent (Bulletproof) --------
router_agent = Agent(
    name="routing_agent",
    model="gemini-2.0-flash",
    description="Generates exactly 10 distinct walking routes using Google Routes API.",
    instruction=(
        "You must call plan_routes(origin, destination, count=10, output='json') with the user's origin and destination. "
        "After calling the function, examine the result. "
        "If status is 'success', reply EXACTLY: 'Route generation complete. Found [X] routes. Passing to safety analysis.' "
        "If status is 'error', reply EXACTLY: 'Route generation failed: [error_message]. Please try different locations.' "
        "Do NOT display the JSON data. Do NOT explain the routes. The next agent will handle analysis."
    ),
    tools=[plan_routes],
)


# -------- Agent 2: ranks supplied routes by crime risk and summarizes to the user --------
# agent.py (crime agent only)
crime_agent = Agent(
    name="crime_route_agent",
    model="gemini-2.0-flash",
    description="Analyzes routes for safety using crime data.",
    instruction=(
        "You have received routing data from the previous agent. "
        "Call parse_and_rank_routes() with the routing context. "
        "Do not use print(). Do not use any API prefix. "
        "Do not make classifications - rank the data you were given."
    ),
    tools=[parse_and_rank_routes, get_route_details],
)
# -------- Pipeline: run Router first, then Crime ranker --------
code_pipeline_agent = SequentialAgent(
    name="SafeRouteAgent",
    sub_agents=[router_agent, crime_agent],
    description=(
        "Two-step process: 1) routing_agent generates routes as JSON, "
        "2) crime_route_agent analyzes those routes for safety. "
        "The routing_agent's output must be passed as context to crime_route_agent. "
        "Always run both agents in sequence - never respond as 'root agent'."
    ),
)



root_agent = code_pipeline_agent  # Use this instead of SequentialAgent


