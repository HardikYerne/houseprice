import requests
import json

# Configuration
BASE_URL = "http://127.0.0.1:5000"

def test_connection():
    """Test all endpoints"""
    print("="*50)
    print("TESTING FLASK API CONNECTION")
    print("="*50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"\n1. Health Check: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ {response.json()}")
        else:
            print(f"   ❌ {response.text}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # Test 2: Simple prediction
    try:
        test_data = {
            "MedInc": 3.5,
            "HouseAge": 29,
            "AveRooms": 5.5,
            "AveBedrms": 1.1,
            "Population": 500,
            "AveOccup": 2.5,
            "Latitude": 35.5,
            "Longitude": -119.5
        }
        
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"\n2. Prediction Test: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ {response.json()}")
        else:
            print(f"   ❌ {response.text}")
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    test_connection()