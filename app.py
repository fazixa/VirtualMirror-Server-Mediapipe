import os
import click
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
# from src.tints.db.database import DB

from src.route.simulation import *
from src.route.eyeshadow import *
from src.route.blush import *
from src.route.foundation import *
from src.route.lipstick import *
from src.route.eyeliner import *
from src.route.concealer import *
from src.route.lens import *


# Load environment config from .env file




# app.register_blueprint(simulation)
# app.register_blueprint(eyeshadow)
# app.register_blueprint(blush)
# app.register_blueprint(foundation)
# app.register_blueprint(lipstick)
# app.register_blueprint(eyeliner)
# app.register_blueprint(concealer)
# app.register_blueprint(lens)

# Start connect to MongoDB
# DB.init()


@click.command()
@click.option("--host", default='0.0.0.0', help="The interface to bind to.")
@click.option("--port", default=5000, help="The port to bind to.")
@click.option("--debug", default=True, help="Set debug mode.")
@click.option("--camera-index", default=0, help="Index of the camera to use in the app.")
def run_server(host, port, debug, camera_index):
    app = Flask(__name__)

    # Register routes in application
    app.register_blueprint(construct_simulation(camera_index))
    app.register_blueprint(eyeshadow)
    app.register_blueprint(blush)
    app.register_blueprint(foundation)
    app.register_blueprint(lipstick)
    app.register_blueprint(eyeliner)
    app.register_blueprint(concealer)
    app.register_blueprint(lens)

    bcrypt = Bcrypt(app)
    jwt = JWTManager(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.config["JWT_SECRET_KEY"] = os.getenv('JWT_SECRET_KEY')
    CORS(app, expose_headers=["x-suggested-filename"],
        resources={r"/*": {"origins": "*"}})
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    run_server()
