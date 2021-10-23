# Virtual Mirror MediaPipe
Backend API for Virtual Mirror Web Application

## Documentation
Documentation can be found [here](/docs/build/html/index.html).

## Requirements
- Python 3.7+
- build-essentials

## How to run server

1. Install the required libraries
    ```
    $ pip install -r requirements.txt
    ```

2. Setup the environment
    ```
    $ export FLASK_APP=server_config
    ```

3. Run the server
    ```
    $ flask run
    ```

## To freeze libraries to requirements.txt
```
$ pip freeze > requirements.txt
```
