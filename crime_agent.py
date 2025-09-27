"""
crime_agent.py
--------------
Minimal crime data agent boilerplate.

Reads mock crime data from data/mock_data.json and returns
a simplified list of {street, city, lat, lon} entries.

Add routing, filtering, or scoring logic later as needed.
"""

import json
from typing import List, Dict
from google.adk.agents import Agent


def get_crime_locations() -> List[Dict[str, str]]:
    """
    Load incidents from data/mock_data.json and return a simple list.

    Returns:
        List of dicts like:
        [
          {"street": "1200 BISCAYNE BLVD", "city": "MIA",
           "lat": 25.7877, "lon": -80.1870},
          ...
        ]
    """
    with open("data/mock_data.json", "r") as f:
        data = json.load(f)

    results = []
    for inc in data.get("incidents", []):
        results.append({
            "street": inc.get("incident_address", ""),
            "city": inc.get("city_key", ""),
            "lat": inc.get("incident_latitude"),
            "lon": inc.get("incident_longitude"),
        })
    return results


# Instantiate the agent so it can be imported directly
crime_agent = Agent(
    name= "crime_data_agent",
    model= "gemini-2.0-flash",
    description= "Returns simple street/city/lat/lon info from mock crime data.",
    instruction= "Call get_crime_locations to retrieve the list of crime locations.",
    tools= [get_crime_locations]
)

if __name__ == "__main__":
    for loc in get_crime_locations()[:5]:
        print(loc)