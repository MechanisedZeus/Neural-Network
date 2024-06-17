import tkinter
from tkinter import *
import classify
import play

#
# creates ui window
#

app = tkinter.Tk()
app.geometry("425x125")
app.title("numerical classifier")

#
# code goes here
#


def get_file():
    file_name = file_input.get(1.0, "end-1c")
    file_name = file_name.strip()
    print(file_name)
    file_input.delete("1.0", "end")
    prediction = classify.classify(file_name)
    print(file_name + " is a: " + str(prediction))
    prediction_text.configure(state='normal')
    prediction_text.delete("1.0", "end")
    prediction_text.delete("1.0", "end")
    prediction_text.insert("end", "The number is a: " + str(prediction))
    prediction_text.configure(state='disabled')
    play.play(prediction)              # plays audio


def enter_key(event):                                   # prevents errors when using enter keys and buttons
    get_file()


#
# creates ui elements
#

file_input = Text(app, height=1, width=40)
prediction_text = Text(app, height=1, width=18)
enter_button = tkinter.Button(app, text="enter", command=get_file)
prediction_text.configure(state='disabled')

app.bind('<Return>', enter_key)                         # detects when the user presses the enter key


#
# displays ui elements
#

enter_button.pack()
file_input.pack()
prediction_text.pack()
app.mainloop()
