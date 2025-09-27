# agent.py
from google.adk.agents import Agent, SequentialAgent

# Tools from agents
from multichatter.utils.agent1_utils import plan_routes
from multichatter.utils.agent2_utils import (
    rank_supplied_routes,
    get_route_details,
    get_crime_locations,   # optional
    get_ranked_routes,     # optional
)

# -------- Agent 1: finds routes and returns machine-readable JSON only --------
# agent.py (router agent only)
router_agent = Agent(
    name="routing_agent",
    model="gemini-2.0-flash",
    description="Finds up to 10 distinct walking routes (machine output only).",
    instruction=(
        "First, call plan_routes(origin, destination, count=10, output='json'). "
        "After the tool returns, do NOT paste the JSON. "
        "Reply to the USER with exactly: 'Got it — generating routes and passing them to safety scoring now.' "
        "The tool result will be available to the next agent in context."
    ),

    tools=[plan_routes],
)


# -------- Agent 2: ranks supplied routes by crime risk and summarizes to the user --------
# agent.py (crime agent only)
crime_agent = Agent(
    name="crime_route_agent",
    model="gemini-2.0-flash",
    description="Scores walking routes using local incident data and summarizes clearly.",
    instruction=(
        "Find the most recent function/tool response from routing_agent with keys "
        "['status','origin','destination','routes']. "
        "If missing or status!='success', apologize briefly and ask the user to rephrase origin/destination. "
        "Otherwise, call rank_supplied_routes(routes=<that['routes']>, origin=<that['origin']>, destination=<that['destination']>). "
        "Then talk to the USER like a normal person: start with 'I generated N routes and ranked their safety.' "
        "Show the TOP 3 with: rank, duration (min), distance (km), risk_summary, and maps_link. "
        "Do NOT display any encoded polylines. "
        "If asked about a route_id, call get_route_details(route_id) and explain key risk factors (still no polyline)."
    ),
    tools=[rank_supplied_routes, get_route_details],
)

# -------- Pipeline: run Router first, then Crime ranker --------
code_pipeline_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[router_agent, crime_agent],
    description="1) Get routes as JSON (machine output). 2) Rank those routes and produce a user-facing safety summary.",
)

# ADK expects the root agent to be named `root_agent`
root_agent = code_pipeline_agent
