from google.adk.agents import Agent, SequentialAgent
from utils.agent1_utils import plan_routes 




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


code_pipeline_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[router_agent],
    description="Executes a sequence of code writing, reviewing, and refactoring.",
    # The agents will run in the order provided: Writer -> Reviewer -> Refactorer
)



# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = code_pipeline_agent