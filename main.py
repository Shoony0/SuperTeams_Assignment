import base64
from fastapi import FastAPI, HTTPException, Request
import requests
from dotenv import load_dotenv
import os
from PIL import Image
import io

app = FastAPI()
load_dotenv(f"{os.getcwd()}/.env", override=True)

# Function to call the Replicate API
def generate_image(prompt: str, count: int):
    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

    body = {
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "cfg_scale": 5,
        "samples": 1,
        "text_prompts": [
            {
            "text": prompt,
            "weight": 1
            }
        ],
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('REPLICATE_API_TOKEN')}",
    }

    response = requests.post(
        url,
        headers=headers,
        json=body,
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    for image in data["artifacts"]:
        base64_string = image["base64"]
        # Decode the Base64 string
        image_bytes = base64.b64decode(base64_string)

        # Create an image object from the decoded bytes
        image = Image.open(io.BytesIO(image_bytes))

        image_file_path = f'{os.getcwd()}/generate_images/{prompt.replace(" ", "_")}_{count}.jpg'

        # Save the image to a file
        image.save(image_file_path, "JPEG")
    
    return data


# API endpoint to generate images
@app.post("/generate-image")
async def generate_image_endpoint(request: Request):
    """
    Generates an image from a prompt using the Replicate API.
    Request Body:
        - **prompt**: Text prompt to generate the image
        - **no_of_images**: Number of image to create
    """
    try:
        request_json = await request.json()
        no_of_images = request_json["no_of_images"]
        result = []
        for index, _ in enumerate(range(no_of_images)):
            output = generate_image(request_json["prompt"], count=index+1)
            result.append(output)
        return {"status": "success", "mesage": "Images Generated Successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
