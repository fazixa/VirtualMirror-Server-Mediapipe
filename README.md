# Virtual Mirror MediaPipe
Backend API for Virtual Mirror Web Application

## Documentation
Documentation can be found [here](/docs/build/html/index.html).

## How to run the server

### Using our docker image (recommended)
1. Clone the repository
    ```
    $ git clone https://github.com/fazixa/VirtualMirror-Server-Mediapipe.git
    ```

2. Change ports, environment variables, and device mapping in the compose file to your liking.

3. Use `docker compose` to run the project
    ```
    $ docker compose up -d
    ```
    
The app can now be accessed at `localhost:8080`.

**Notes**:
- By running the app using docker, the frontend module will also be deployed. If you choose to run the project without docker, you should run the frontend module independantly. Please check that module at [VirtualMirror-Frontend](https://github.com/fazixa/VirtualMirror-Frontend)
- Users working with WSL2 may struggle with accessing USB cameras and mapping devices like `/dev/video*`. You can consult [this repository](https://github.com/dorssel/usbipd-win/wiki/WSL-support) for possible solutions.

### Local installation
#### Requirements
- Python 3.7+
- build-essentials
#### Steps
1. Install the required libraries
    ```
    $ pip install -r requirements.txt
    ```
**Note**: Building dlib may take long. Let the build run and do not interrupt the process.

2. Run the server
    ```
    $ python app.py --host 0.0.0.0 --port 5000 --debug True --camera-index 0
    ```
