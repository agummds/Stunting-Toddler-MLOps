import requests
import json

# Test URL
url = "http://10.136.165.49:5000/predict"

print("=== Testing Stunting Detection API ===")

# Test data examples with proper format
test_cases = [
    {
        "name": "Anak Normal",
        "input": {
            "Umur (bulan)": 24,
            "Jenis Kelamin": 1,  # 1 = Laki-laki
            "Tinggi Badan (cm)": 87.5
        }
    },
    {
        "name": "Anak Berpotensi Stunting",
        "input": {
            "Umur (bulan)": 36,
            "Jenis Kelamin": 0,  # 0 = Perempuan
            "Tinggi Badan (cm)": 82.0
        }
    },
    {
        "name": "Anak Muda Normal",
        "input": {
            "Umur (bulan)": 12,
            "Jenis Kelamin": 1,  # 1 = Laki-laki
            "Tinggi Badan (cm)": 75.0
        }
    }
]

print("Testing GET request first...")
try:
    get_response = requests.get(url)
    print(f"GET Response: {get_response.json()}")
except Exception as e:
    print(f"GET Error: {e}")

print("\n" + "="*60)

for i, test_case in enumerate(test_cases, 1):
    print(f"\nTest Case {i}: {test_case['name']}")
    print(f"Input: {test_case['input']}")
    print("-" * 40)
    
    try:
        response = requests.post(url, json={"input": test_case['input']})
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction: {result['prediction']}")
            print(f"Status: {result['status']}")
        else:
            print(f"❌ Error: {response.json()}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure Flask server is running.")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*60)
print("Testing health endpoint...")
try:
    health_response = requests.get("http://10.136.165.49:5000/health")
    print(f"Health Status: {health_response.json()}")
except Exception as e:
    print(f"Health check error: {e}")
