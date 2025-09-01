#!/usr/bin/env python3

import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

def check_apis():
    credentials_path = '/home/luker/work/cara/analytics/data_ingestion/google_api_luker.json'
    
    with open(credentials_path, 'r') as f:
        credentials_info = json.load(f)
    
    credentials = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=['https://www.googleapis.com/auth/cloud-platform.read-only']
    )
    
    try:
        service = build('serviceusage', 'v1', credentials=credentials)
        project_id = credentials_info['project_id']
        
        request = service.services().list(parent=f'projects/{project_id}')
        response = request.execute()
        
        analytics_apis = [s for s in response.get('services', []) 
                         if 'analytics' in s['config']['name'].lower()]
        
        print("Analytics-related APIs:")
        for api in analytics_apis:
            name = api['config']['name']
            state = api['state']
            print(f"  {name}: {state}")
            
    except Exception as e:
        print(f"Cannot check APIs: {e}")
        print("This is normal - try the Airbyte connection anyway")

if __name__ == "__main__":
    check_apis()