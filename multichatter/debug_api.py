# Create debug_api.py
import os
import httpx

def test_api_setup():
    """Test your Google API setup"""
    
    maps_key = os.getenv("MAPS_KEY", "")
    print(f"🔑 API Key exists: {bool(maps_key)}")
    print(f"🔑 API Key length: {len(maps_key)}")
    print(f"🔑 API Key starts with: {maps_key[:10]}..." if maps_key else "No key found")
    
    if not maps_key:
        print("❌ MAPS_KEY environment variable not set!")
        print("Set it with: export MAPS_KEY=your_api_key_here")
        return False
    
    return True

def test_simple_api_call():
    """Test a simple API call to see what error we get"""
    
    maps_key = os.getenv("MAPS_KEY", "")
    if not maps_key:
        print("No API key found")
        return
    
    # Test the simplest possible request
    headers = {
        "X-Goog-Api-Key": maps_key,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline",
        "Content-Type": "application/json"
    }
    
    # Simple request body
    body = {
        "origin": {"address": "Miami Beach, FL"},
        "destination": {"address": "Downtown Miami, FL"},
        "travelMode": "WALK"
    }
    
    print("🌐 Testing API call...")
    print(f"URL: https://routes.googleapis.com/directions/v2:computeRoutes")
    print(f"Headers: {headers}")
    print(f"Body: {body}")
    
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "https://routes.googleapis.com/directions/v2:computeRoutes",
                json=body,
                headers=headers
            )
        
        print(f"✅ Status Code: {resp.status_code}")
        print(f"📥 Response: {resp.text[:500]}...")
        
        if resp.status_code != 200:
            print(f"❌ Error: {resp.status_code}")
            print(f"Full response: {resp.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    if test_api_setup():
        test_simple_api_call()