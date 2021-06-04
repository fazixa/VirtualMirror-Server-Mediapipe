import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
# from src.tints.db.database import DB

from src.route.simulation import *
from src.route.eyeshadow import *
# from src.route.blush import *
# from src.route.foundation import *
from src.route.lipstick import *
from src.route.eyeliner import *
# from src.route.concealer import *

# Load environment config from .env file
load_dotenv()
# App reference
app = Flask(__name__)
app.debug = True
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["JWT_SECRET_KEY"] = os.getenv('JWT_SECRET_KEY')
CORS(app, expose_headers=["x-suggested-filename"],
     resources={r"/*": {"origins": "*"}})
host = os.environ.get('IP', '0.0.0.0')
port = int(os.environ.get('PORT', 5000))


# Register route in application


app.register_blueprint(simulation)
app.register_blueprint(eyeshadow)
# app.register_blueprint(blushr)
# app.register_blueprint(foundationm)
app.register_blueprint(lipstick)
app.register_blueprint(eyeliner)
# app.register_blueprint(concealerm)

# Start connect to MongoDB
# DB.init()

if __name__ == "__main__":
    app.run(host=host, port=port)
