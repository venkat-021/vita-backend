from urllib.parse import urlencode
import webbrowser
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from typing import Optional, Dict, List
import asyncio
from bleak import BleakScanner, BleakClient
import time
from dataclasses import dataclass
from enum import Enum
import struct
import sys

class WatchBrand(Enum):
    XIAOMI = "xiaomi"
    REALME = "realme"
    NOTHING = "nothing"
    FIREBOLTT = "fireboltt"
    BOAT = "boat"
    UNKNOWN = "unknown"

@dataclass
class WatchData:
    steps: int
    heart_rate: int
    battery_level: int
    timestamp: float
    brand: WatchBrand
    # Nothing Watch specific fields
    calories: int = 0
    distance: float = 0.0
    sleep_duration: int = 0
    stress_level: int = 0

class OAuth2Handler:
    def __init__(self):
        self.client_id = "YOUR_CLIENT_ID"  # Replace with your actual client ID
        self.client_secret = "YOUR_CLIENT_SECRET"  # Replace with your actual client secret
        self.redirect_uri = "http://localhost:8000/callback"
        self.scope = "https://www.googleapis.com/auth/fitness.activity.read"
        self.token = None

    def get_auth_url(self) -> str:
        """Generate the authorization URL for Google OAuth2."""
        auth_params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': self.scope,
            'access_type': 'offline',
            'prompt': 'consent'
        }
        return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(auth_params)}"

    def exchange_code_for_token(self, code: str) -> Optional[dict]:
        """Exchange the authorization code for access and refresh tokens."""
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error exchanging code for token: {e}")
            return None

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle the OAuth2 callback."""
        if self.path.startswith('/callback'):
            # Extract the authorization code from the URL
            query_components = self.path.split('?')[1].split('&')
            code = None
            for component in query_components:
                if component.startswith('code='):
                    code = component.split('=')[1]
                    break

            if code:
                # Exchange the code for tokens
                oauth_handler = OAuth2Handler()
                token_data = oauth_handler.exchange_code_for_token(code)
                
                if token_data:
                    # Save the tokens securely (in a real application, use proper secure storage)
                    with open('token_data.json', 'w') as f:
                        json.dump(token_data, f)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"Authentication successful! You can close this window.")
                else:
                    self.send_error(500, "Failed to exchange code for token")
            else:
                self.send_error(400, "No authorization code received")
        else:
            self.send_error(404, "Not found")

class SmartWatchManager:
    def __init__(self):
        self.connected_watches: Dict[str, WatchData] = {}
        self.supported_services = {
            WatchBrand.XIAOMI: {
                "service_uuid": "0000fee0-0000-1000-8000-00805f9b34fb",
                "characteristic_uuid": "0000fee1-0000-1000-8000-00805f9b34fb"
            },
            WatchBrand.REALME: {
                "service_uuid": "0000fee0-0000-1000-8000-00805f9b34fb",
                "characteristic_uuid": "0000fee1-0000-1000-8000-00805f9b34fb"
            },
            WatchBrand.NOTHING: {
                # CMF Watch Pro 2 specific UUIDs
                "service_uuid": "0000ffd0-0000-1000-8000-00805f9b34fb",
                "data_characteristic": "0000ffd1-0000-1000-8000-00805f9b34fb",
                "notification_characteristic": "0000ffd2-0000-1000-8000-00805f9b34fb",
                "control_characteristic": "0000ffd3-0000-1000-8000-00805f9b34fb"
            }
        }
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.client = None  # Store the client instance

    async def check_bluetooth(self) -> bool:
        """Check if Bluetooth is available and enabled."""
        try:
            scanner = BleakScanner()
            await scanner.start()
            await scanner.stop()
            return True
        except OSError as e:
            if "device is not ready" in str(e).lower():
                print("\nError: Bluetooth adapter is not ready or not enabled.")
                print("Please make sure:")
                print("1. Your computer has Bluetooth capability")
                print("2. Bluetooth is turned on")
                print("3. You have the necessary permissions")
                print("\nOn Windows, you can enable Bluetooth by:")
                print("1. Opening Settings")
                print("2. Going to Bluetooth & devices")
                print("3. Turning on Bluetooth")
            else:
                print(f"\nError checking Bluetooth: {e}")
            return False
        except Exception as e:
            print(f"\nUnexpected error checking Bluetooth: {e}")
            return False

    async def scan_for_watches(self) -> List[Dict]:
        """Scan for nearby smartwatches."""
        if not await self.check_bluetooth():
            return []

        print("Scanning for nearby smartwatches...")
        try:
            devices = await BleakScanner.discover()
            
            watches = []
            for device in devices:
                brand = self._identify_brand(device.name or "")
                if brand != WatchBrand.UNKNOWN:
                    watches.append({
                        "address": device.address,
                        "name": device.name,
                        "brand": brand
                    })
            return watches
        except Exception as e:
            print(f"Error scanning for watches: {e}")
            return []

    def _identify_brand(self, device_name: str) -> WatchBrand:
        """Identify the watch brand from device name."""
        device_name = device_name.lower()
        if "mi band" in device_name or "xiaomi" in device_name:
            return WatchBrand.XIAOMI
        elif "realme" in device_name:
            return WatchBrand.REALME
        elif "nothing" in device_name or "cmf" in device_name:  # Nothing Watch uses CMF branding
            return WatchBrand.NOTHING
        elif "fireboltt" in device_name:
            return WatchBrand.FIREBOLTT
        elif "boat" in device_name:
            return WatchBrand.BOAT
        return WatchBrand.UNKNOWN

    async def connect_to_watch(self, address: str, brand: WatchBrand) -> bool:
        """Connect to a specific smartwatch."""
        for attempt in range(self.max_retries):
            try:
                print(f"\nAttempting to connect to {brand.value} watch (attempt {attempt + 1}/{self.max_retries})...")
                
                # Create client with security level
                client = BleakClient(
                    address,
                    security_level="high"  # Use high security level for CMF Watch
                )
                
                # Connect with encryption
                await client.connect()
                
                # Initialize Nothing Watch if needed
                if brand == WatchBrand.NOTHING:
                    if not await self._initialize_nothing_watch(client):
                        print("Failed to initialize Nothing Watch")
                        await client.disconnect()
                        continue
                
                # Store connection
                self.connected_watches[address] = WatchData(
                    steps=0,
                    heart_rate=0,
                    battery_level=0,
                    timestamp=time.time(),
                    brand=brand
                )
                
                # Store client instance
                self.client = client
                
                print(f"Successfully connected to {brand.value} watch at {address}")
                return True
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    print("\nTroubleshooting tips:")
                    print("1. Make sure the watch is in pairing mode")
                    print("2. Keep the watch close to your computer")
                    print("3. Try restarting the watch")
                    print("4. Make sure no other device is connected to the watch")
                    print("5. Check if your system supports BLE encryption")
                    print("6. Try unpairing and repairing the watch")
                    return False

    async def _initialize_nothing_watch(self, client: BleakClient) -> bool:
        """Initialize Nothing Watch with proper sequence."""
        try:
            print("Initializing Nothing Watch...")
            
            # Store the client instance
            self.client = client
            
            # Wait for encryption to be established
            await asyncio.sleep(2)
            
            # Step 1: Enable notifications
            print("Enabling notifications...")
            await client.start_notify(
                self.supported_services[WatchBrand.NOTHING]["notification_characteristic"],
                self._notification_handler
            )
            
            # Step 2: Send initialization command
            print("Sending initialization command...")
            init_command = bytes([0x01, 0x00, 0x00, 0x00])  # Example initialization command
            await client.write_gatt_char(
                self.supported_services[WatchBrand.NOTHING]["control_characteristic"],
                init_command,
                response=True
            )
            
            # Wait for initialization to complete
            await asyncio.sleep(1)
            
            # Step 3: Enable data notifications
            print("Enabling data notifications...")
            await client.write_gatt_char(
                self.supported_services[WatchBrand.NOTHING]["control_characteristic"],
                bytes([0x01]),
                response=True
            )
            
            print("Nothing Watch initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing Nothing Watch: {e}")
            return False

    def _notification_handler(self, sender, data):
        """Handle notifications from the watch."""
        try:
            print(f"Received notification: {data.hex()}")
            # Add notification parsing logic here if needed
        except Exception as e:
            print(f"Error handling notification: {e}")

    async def _read_nothing_watch_data(self, client: BleakClient) -> Optional[WatchData]:
        """Read data specifically from Nothing Watch."""
        for attempt in range(self.max_retries):
            try:
                print(f"\nReading data from Nothing Watch (attempt {attempt + 1}/{self.max_retries})...")
                
                # Ensure we're still connected
                if not client.is_connected:
                    print("Connection lost, attempting to reconnect...")
                    await client.connect()
                    if not await self._initialize_nothing_watch(client):
                        raise Exception("Failed to reinitialize watch")
                
                # Request data update
                print("Requesting data update...")
                await client.write_gatt_char(
                    self.supported_services[WatchBrand.NOTHING]["control_characteristic"],
                    bytes([0x02]),
                    response=True
                )
                
                # Wait for data to be ready
                await asyncio.sleep(2)
                
                # Read data
                print("Reading data...")
                metrics_data = await client.read_gatt_char(
                    self.supported_services[WatchBrand.NOTHING]["data_characteristic"],
                    response=True
                )
                
                # Parse the data (this is an example structure, actual implementation may vary)
                battery_level = int.from_bytes(metrics_data[0:1], byteorder='little')
                steps = int.from_bytes(metrics_data[1:5], byteorder='little')
                heart_rate = int.from_bytes(metrics_data[5:6], byteorder='little')
                calories = int.from_bytes(metrics_data[6:10], byteorder='little')
                distance = struct.unpack('<f', metrics_data[10:14])[0]  # float in meters
                sleep_duration = int.from_bytes(metrics_data[14:18], byteorder='little')  # in minutes
                stress_level = int.from_bytes(metrics_data[18:19], byteorder='little')

                print("Successfully read watch data!")
                print(f"Battery level: {battery_level}%")
                print(f"Steps: {steps}")
                print(f"Heart Rate: {heart_rate} bpm")
                print(f"Calories: {calories}")
                print(f"Distance: {distance:.2f} meters")
                print(f"Sleep Duration: {sleep_duration} minutes")
                print(f"Stress Level: {stress_level}")

                return WatchData(
                    steps=steps,
                    heart_rate=heart_rate,
                    battery_level=battery_level,
                    timestamp=time.time(),
                    brand=WatchBrand.NOTHING,
                    calories=calories,
                    distance=distance,
                    sleep_duration=sleep_duration,
                    stress_level=stress_level
                )
            except Exception as e:
                print(f"Error reading data (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    print("\nTroubleshooting tips:")
                    print("1. Make sure the watch is still connected")
                    print("2. Try reconnecting to the watch")
                    print("3. Check if the watch's battery is not too low")
                    print("4. Try restarting the watch")
                    return None

    async def read_watch_data(self, address: str) -> Optional[WatchData]:
        """Read data from a connected watch."""
        if address not in self.connected_watches:
            print(f"No watch connected at address {address}")
            return None

        try:
            # Use the stored client instance if available
            client = self.client if self.client else BleakClient(address)
            
            if not client.is_connected:
                await client.connect()
                if self.connected_watches[address].brand == WatchBrand.NOTHING:
                    await self._initialize_nothing_watch(client)

            watch_data = self.connected_watches[address]
            brand = watch_data.brand
            
            if brand == WatchBrand.NOTHING:
                return await self._read_nothing_watch_data(client)
            elif brand in self.supported_services:
                service = self.supported_services[brand]
                # Read data from the watch (implementation would vary by brand)
                watch_data.timestamp = time.time()
                return watch_data
            else:
                print(f"Unsupported watch brand: {brand}")
                return None
        except Exception as e:
            print(f"Error reading watch data: {e}")
            return None

    def save_data_to_file(self, data: WatchData, filename: str = "watch_data.json"):
        """Save watch data to a JSON file."""
        try:
            data_dict = {
                "steps": data.steps,
                "heart_rate": data.heart_rate,
                "battery_level": data.battery_level,
                "timestamp": data.timestamp,
                "brand": data.brand.value
            }
            
            # Add Nothing Watch specific fields if available
            if data.brand == WatchBrand.NOTHING:
                data_dict.update({
                    "calories": data.calories,
                    "distance": data.distance,
                    "sleep_duration": data.sleep_duration,
                    "stress_level": data.stress_level
                })
            
            with open(filename, 'w') as f:
                json.dump(data_dict, f, indent=4)
            print(f"\nData saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")

async def main():
    print("Smart Watch Data Reader")
    print("======================")
    
    manager = SmartWatchManager()
    
    # Scan for watches
    watches = await manager.scan_for_watches()
    
    if not watches:
        print("\nNo compatible watches found")
        print("Please make sure:")
        print("1. Your watch is turned on")
        print("2. Your watch is in pairing mode")
        print("3. Your watch is close to your computer")
        return
    
    print("\nFound watches:")
    for i, watch in enumerate(watches, 1):
        print(f"{i}. {watch['name']} ({watch['brand'].value})")
    
    # Connect to the first watch found
    if watches:
        watch = watches[0]
        if await manager.connect_to_watch(watch['address'], watch['brand']):
            # Read data
            data = await manager.read_watch_data(watch['address'])
            if data:
                manager.save_data_to_file(data)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please try again or contact support if the problem persists")
