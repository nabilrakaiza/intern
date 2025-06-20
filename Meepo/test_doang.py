import json
import os
from typing import Any

import requests


def render_graphic_design(
        json_data: dict[str, Any],
        id_template: str
    ) -> tuple[dict[str, Any], str]:
        '''render a template with modified field on json_data'''
        url = f'{os.getenv("MEEPO_TEMPLATE_BACKEND_URL")}/template/{id_template}/fill'
        
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        responses = requests.post(url, headers={
            'content-types': 'application/json'
        }, json=json_data)
        # Convert JSON string to a Python dictionary
        response_data = json.loads(responses.text)
        print("response_data 557", response_data)
        data = response_data.get("data", "https://meepo-new.s3.ap-southeast-1.amazonaws.com/meepo_template/assets/output_20250313_062836.png") 
        gd_id = response_data.get("gd_id", "error")
        return data, gd_id

s = {"BG": "https://replicate.com/p/azp87hrfphrmc0cqbqavv5a2p4",
     "IMRB": "https://replicate.com/p/360qhhrj15rmc0cqbqabdk3q3w"}

render_graphic_design(s, "681d4b6e42b8ba81a27b9c7d")