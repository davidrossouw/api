import cv2
import numpy as np

from fastapi import FastAPI, File
from pydantic import BaseModel

from model import SimpsonClassifier	


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


@app.post("/classify_image/")
async def create_file(file: bytes = File(...)):
	# CV2
	img_size = 64
	img = cv2.imdecode(np.fromstring(file, np.uint8), cv2.IMREAD_COLOR)
	img = cv2.resize(img, (img_size, img_size)).astype('float32') / 255.
	img = np.expand_dims(img, axis=0)
	
	model = SimpsonClassifier(weights_path='/Users/davidrossouw/Documents/my_projects/api/fastapi/app/data/weights.best.hdf5', pic_size=img_size)
	result = model.run(img)
	print(result)
	
	return {"y_pred": result['y_pred'], "y_prob": result['y_prob']}
