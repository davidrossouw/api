# Simpson classifier Service

from flask import Flask
from flask_restful import Resource, Api, reqparse
import numpy as np
import cv2
import time
from model import SimpsonClassifier	

import werkzeug, os

# Instantiate the app
app = Flask(__name__)
api = Api(app)

UPLOAD_FOLDER = 'static/img'
parser = reqparse.RequestParser()
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')

def read_image(file):

    #read image file string data
    filestr = file.read()
    #convert string data to numpy array
    print(type(filestr))

    nparr = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class PhotoUpload(Resource):
    decorators=[]

    def post(self):
        data = parser.parse_args()
        if data['file'] == "":
            return {
                    'data':'',
                    'message':'No file found',
                    'status':'error'
                    }
        photo = data['file']
        print(type(photo))
        if photo:
            filename = 'your_image.png'
            #photo.save(os.path.join(UPLOAD_FOLDER,filename))

            # read image file 
            img = read_image(photo)
            img_size = 64
            img = cv2.resize(img, (img_size, img_size)).astype('float32') / 255.
            img = np.expand_dims(img, axis=0)

            # Instantiate model
            model = SimpsonClassifier(weights_path='./data/weights.best.hdf5', pic_size=img_size)
            
            # Run model
            result = model.run(img)	


            return {
                    'data':'',
                    'message':'photo uploaded',
                    'status':'success',
                    'image size': img.shape

                    }
        return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }


# import jsonpickle
# import numpy as np
# import cv2


# # route http posts to this method
# @app.route('/api/test', methods=['POST'])
# def test():
#     r = request
#     # convert string of image data to uint8
#     nparr = np.fromstring(r.data, np.uint8)
#     # decode image
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # do some fancy processing here....

#     # build a response dict to send back to client
#     response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
#                 }
#     # encode response using jsonpickle
#     response_pickled = jsonpickle.encode(response)

#     return Response(response=response_pickled, status=200, mimetype="application/json")



api.add_resource(HelloWorld, '/')
api.add_resource(PhotoUpload,'/upload')

if __name__ == '__main__':
    app.run(debug=True)












# @app.post("/items/")
# async def create_item(item: Item):
# 	item_dict = item.dict()
# 	if item.tax:
# 		price_with_tax = item.price + item.tax
# 		item_dict.update({"price_with_tax": price_with_tax})
# 	return item_dict


# @app.post("/classify_image/")
# async def classify_image(file: bytes = File(...)):
# 	# Prepare image
# 	img_size = 64
# 	img = cv2.imdecode(np.fromstring(file, np.uint8), cv2.IMREAD_COLOR)
# 	img = cv2.resize(img, (img_size, img_size)).astype('float32') / 255.
# 	img = np.expand_dims(img, axis=0)
	
# 	# Instantiate model
# 	model = SimpsonClassifier(weights_path='./data/weights.best.hdf5', pic_size=img_size)
	
# 	# Run model
# 	result = model.run(img)	
	
# 	return {"y_pred": result['y_pred'], "y_prob": result['y_prob']}








# Create routes
api.add_resource(Product, '/')

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
