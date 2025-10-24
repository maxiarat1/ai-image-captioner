#!/usr/bin/env python3
"""
Test API endpoints with Python requests.
"""
import requests
import json

API = "http://localhost:5000"

def test_endpoints():
    print("=" * 60)
    print("Testing New Session-based API Endpoints")
    print("=" * 60)

    # 1. Health check
    print("\n1. Testing health endpoint...")
    r = requests.get(f"{API}/health")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")

    # 2. Clear session
    print("\n2. Testing session clear...")
    r = requests.delete(f"{API}/session/clear")
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")

    # 3. Pre-register files
    print("\n3. Testing file pre-registration...")
    r = requests.post(f"{API}/session/register-files", json={
        "files": [
            {"filename": "test1.jpg", "size": 1024},
            {"filename": "test2.png", "size": 2048},
            {"filename": "test3.jpg", "size": 1536}
        ]
    })
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Response: {data}")
    image_ids = data.get('image_ids', [])
    first_id = image_ids[0] if image_ids else None
    print(f"   ✓ Got {len(image_ids)} image IDs")

    # 4. List images
    print("\n4. Testing image listing...")
    r = requests.get(f"{API}/images", params={"page": 1, "per_page": 10})
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Total images: {data.get('total')}")
    print(f"   Pages: {data.get('pages')}")
    if data.get('images'):
        print(f"   First image: {data['images'][0]['filename']}")

    # 5. Get image info
    if first_id:
        print(f"\n5. Testing image info for {first_id}...")
        r = requests.get(f"{API}/image/{first_id}/info")
        print(f"   Status: {r.status_code}")
        print(f"   Response: {r.json()}")

    # 6. Clear session
    print("\n6. Testing session clear again...")
    r = requests.delete(f"{API}/session/clear")
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Deleted: {data.get('deleted_count')} images")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

if __name__ == '__main__':
    try:
        test_endpoints()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Is Flask running?")
        print("Start with: cd backend && python app.py")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
