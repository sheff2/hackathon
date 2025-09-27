from google.adk.agents import Agent, SequentialAgent
from utils.agent1_utils import plan_routes 
from utils.agent2_utils import get_crime_locations, get_ranked_routes, get_route_details




router_agent = Agent(
    name="routing_agent",
    model="gemini-2.0-flash",
    description="Single-call tool that returns up to 10 distinct walking routes as clickable Google Maps links.",
    instruction=(
        "Call plan_routes(origin, destination, count) once when the user provides start and end points. "
        "Return the tool output exactly as-is (Markdown)."
    ),
    tools=[plan_routes],
)

crime_agent = Agent(
    name="crime_route_agent",
    model="gemini-2.0-flash",
    description="Scores late-night walking routes using nearby crime data.",
    instruction="Use get_ranked_routes to analyze candidate paths and get_route_details for follow-ups.",
    tools=[get_crime_locations, get_ranked_routes, get_route_details],
)

code_pipeline_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[router_agent, crime_agent],
    description="Use the router agent to get the polylines and then give that to the crime agent.",
    # The agents will run in the order provided: Writer -> Reviewer -> Refactorer
)



# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = code_pipeline_agent