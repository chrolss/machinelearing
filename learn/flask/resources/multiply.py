from flask_restful import Resource
from flask import request

class multiply(Resource):
    def get(self):
        # curl http://localhost:5000/api/multiply -d "data=hejsan" -X GET
        tempVar = request.form['data']
        return "You said: " + tempVar

    def put(self):
        var = request.form['data']
        return var*2

