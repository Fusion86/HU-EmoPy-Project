import time

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

from fer_client import FERClient


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Emotion Detection Demo")
        self.minsize(640, 400)

        self.button = ttk.Button(
            self, text="Browse A File", command=self.fileDialog)
        self.button.grid(column=0, row=0)

        self.lblHost = Label(self, text="Host")
        self.lblHost.grid(column=0, row=1)

        self.host = Entry(self)
        self.host.insert(0, "https://fer.cerbus.nl")  # Default value
        self.host.grid(column=0, row=2)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(title="Select image")

        # Load image
        self.imgRaw = Image.open(self.filename)

        # Calc size with respect to the aspect ratio
        x = self.imgRaw.width / 400
        y = self.imgRaw.height / 400
        scale = min(x, y)

        # Upscale small images, this is a hack. I'm sure there is an easier way to do this.
        if (scale < 1):
            x = 400 / self.imgRaw.width
            y = 400 / self.imgRaw.height
            scale = min(x, y)

        size_x = int(self.imgRaw.width * scale)
        size_y = int(self.imgRaw.height * scale)

        # Resize
        self.imgRaw = self.imgRaw.resize((size_x, size_y), Image.ANTIALIAS)

        # Clear old image
        if hasattr(self, "img"):
            self.img.destroy()

        # Display new image
        self.imgRender = ImageTk.PhotoImage(self.imgRaw)
        self.img = Label(self, image=self.imgRender,
                         height=size_y, width=size_x)
        self.img.image = self.imgRender  # Yes this is needed
        self.img.grid(column=1, row=0)

        # Load emotions from server
        client = FERClient(self.host.get())

        start = time.time_ns()
        res = client.get_emotion(self.filename)
        end = time.time_ns()

        # Build text

        txt = """Top emotions:
{} {}
{} {}
{} {}

Runtime:
Resizing {:.2f}ms
Grayscaling {:.2f}ms
Model {:.2f}ms
Network {:.2f}ms""".format(res["emotions"][0], res["probabilities"][0],
                           res["emotions"][1], res["probabilities"][1],
                           res["emotions"][2], res["probabilities"][2],
                           res["runtime"]["resize"], res["runtime"]["grayscale"], res["runtime"]["model"],
                           (end-start) / 1_000_000)

        # Display emotions
        self.lblRes = Label(self, text=txt)
        self.lblRes.grid(column=1, row=1)


root = Root()
root.mainloop()
