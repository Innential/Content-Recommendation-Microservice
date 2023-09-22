from fastapi import FastAPI
from fastapi import HTTPException
from fastapi_utils.tasks import repeat_every
from pydantic import BaseModel
import requests
import traceback

class Innential(BaseModel):
    skills: list

@repeat_every(seconds=24*60*60)  # 24 hours
async def innential_skills():
    try:
        skills = requests.get("https://api.innential.com/scraper/skills-categories/public")
        skills.raise_for_status()
        json_data = skills.json()

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail="An error occurred and has been saved.")


    subcategories = [list(subcategory_dict.keys())[0] for subcategory_list in json_data.values() for subcategory_dict in
                     subcategory_list]

    Innential.skills = subcategories