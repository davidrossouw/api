from flask import Flask
from flask_restful import Resource, Api, request

app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class PostExample(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        message = json_data['message']
        print(message)

        return message+'api added text'


api.add_resource(HelloWorld, '/api')
api.add_resource(PostExample, '/api/messages')

if __name__ == '__main__':
    app.run(debug=True)
