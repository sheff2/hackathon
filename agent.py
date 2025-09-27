
from google.adk.agents import Agent, SequentialAgent
from . import agent
from utils import agent1_utils, agent2_utils, agent3_utils




crime_agent = Agent(
    name="crime_route_agent",
    model="gemini-2.0-flash",
    description="Scores late-night walking routes using nearby crime data.",
    instruction= "Use get_ranked_routes to analyze candidate paths and get_route_details for follow-ups.",
    tools=[agent2_utils.get_crime_locations(), agent2_utils.get_ranked_routes(), agent2_utils.get_route_details()],
)



# --- 2. Create the SequentialAgent ---
# This agent orchestrates the pipeline by running the sub_agents in order.
code_pipeline_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[code_writer_agent, crime_agent, code_refactorer_agent],
    description="Executes a sequence of code writing, reviewing, and refactoring.",
    # The agents will run in the order provided: Writer -> Reviewer -> Refactorer
)

# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = code_pipeline_agent