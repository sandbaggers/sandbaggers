from flask import Flask
import thread_manager
import threading
from worker import Worker
from endpoints import endpoints

app = Flask(__name__)

if __name__ == "__main__":
    app.run(host='0.0.0.0')

if __name__ == "__main__":
    worker = Worker(target=thread_manager.run, name ="App")

    t = threading.Thread(target=worker.start)
    t.daemon = True
    t.start()

    app.register_blueprint(endpoints)
    app.run(host='0.0.0.0', debug = False)
