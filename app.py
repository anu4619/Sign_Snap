from flask import Flask, render_template, jsonify
import subprocess

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/index2.html")
def index2():
    return render_template("index2.html")

# New route to start the camera and audio detection
@app.route("/start_camera", methods=["POST"])
def start_camera():
    try:
        # Run the audio.py script
        subprocess.Popen(['python', 'Backend/audio.py'])  # Non-blocking call to run the script
        return jsonify({"status": "success", "message": "Camera started successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
