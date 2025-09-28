from utils.agent1_utils import plan_routes
from utils.agent2_utils import debug_agent_context, rank_supplied_routes
import json

def test_individual_functions():
    """Test each function individually"""
    print("🧪 Testing Agent1 function...")
    
    # Test routing function
    routes_result = plan_routes("Miami Beach", "Downtown Miami", count=10)
    print("Agent1 Result:")
    print(json.dumps(routes_result, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Test what happens when we pass this to debug function
    print("🧪 Testing what Agent2 would see...")
    debug_result = debug_agent_context(json.dumps(routes_result))
    
    print("\n" + "="*50 + "\n")
    
    # Test ranking function directly
    if routes_result.get("status") == "success":
        print("🧪 Testing ranking function...")
        ranking_result = rank_supplied_routes(
            routes=routes_result["routes"],
            origin=routes_result["origin"],
            destination=routes_result["destination"]
        )
        
        # Print ALL routes ranked 1-10
        print(f"\n🏆 RANKED ROUTES (1-{len(ranking_result['routes'])})")
        for i, route in enumerate(ranking_result["routes"], 1):
            duration_min = route.get("duration_seconds", 0) / 60
            distance_km = route.get("distance_meters", 0) / 1000
            print(f"{i:2d}. Duration: {duration_min:.1f}min | "
                  f"Distance: {distance_km:.2f}km | "
                  f"Risk: {route.get('risk_score', 0):.4f} | "
                  f"Summary: {route.get('risk_summary', 'N/A')[:50]}...")

if __name__ == "__main__":
    test_individual_functions()