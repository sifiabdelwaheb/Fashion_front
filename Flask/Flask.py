from flask import Flask,jsonify,request
from flask_restful import Api,Resource
from pymongo import MongoClient
app = Flask(__name__)
api = Api(app)
client=MongoClient("mongodb://localhost:27017")
db=client.SentencesDatabase
users=db["users"]

def verify_pw(username,password):
    pass_wd=users.find({
        "username":username
    })[0]['password']
    if pass_wd==password:
        return True
    else:
        return False

def count_token (username):
    tokens=users.find({
        "username":username
    })[0]['token']
    return tokens
class Register(Resource):
    def  post(self):
       postedData=request.get_json()
       username=postedData['username']
       password=postedData['password']

       
       users.insert({
           "username":username,
           "password":password,
           "sentence":"",
           "token":5
       }) 
       retJson={
           "status":200,
           "message":"Succesfuly signed up for this api",
           "response":{
               "username":username,
               "password":password
           }
       }

       return jsonify(retJson)
class Store(Resource):
    def post(self):
        postedData=request.get_json()

        username=postedData['username']
        password=postedData['password']
        sentence=postedData['sentence']

        #Step verify the username and pw match
        correct_pw=verify_pw(username,password)

        if not correct_pw:
            retJson={
                "status":302,

            }
            return retJson
        #Step vetify token
        number_token=count_token(username)
        if number_token <=0:
            retJson={
                "status":302,
                "numbre of token":number_token
            }
            return jsonify(retJson)

        users.update({
            "username":username,

        },{
            "$set":{
                "sentence":sentence,
                "token":number_token-1

            }
        })
        retJson={
            "status":200,
            "msg":"sentence saved successfuly",
            "sentence":sentence,
            "token":number_token-1
           
        }
        return jsonify(retJson)

    

  
api.add_resource(Register,'/register')
api.add_resource(Store,'/store')

@app.route('/')
def helloword():
    return 'hello world'


if __name__ == "__main__":
    app.run(debug=True)
