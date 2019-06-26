import cv2
import numpy as np

from fastapi import FastAPI, File
from pydantic import BaseModel


class Item(BaseModel):
	name: str
	description: str = None
	price: float
	tax: float = None

app = FastAPI()


@app.get("/")
def read_root():
	return {"G'day": "Mate!"}


@app.post("/items/")
async def create_item(item: Item):
	item_dict = item.dict()
	if item.tax:
		price_with_tax = item.price + item.tax
		item_dict.update({"price_with_tax": price_with_tax})
	return item_dict


@app.post("/files/")
async def create_file(file: bytes = File(...)):
	# CV2
	nparr = np.fromstring(file, np.uint8)
	img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	return {"file_size": len(file), "image_size": img_np.shape}