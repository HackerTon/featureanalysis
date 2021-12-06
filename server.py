from quart import Quart, render_template, request, send_file
from quart_cors import cors


app = Quart(__name__)
app = cors(app)


@app.route("/get")
async def get(name):
    tredis = "abc"


if __name__ == "__main__":
    app.run("0.0.0.0")