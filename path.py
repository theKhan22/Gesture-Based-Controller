import tkinter as tk


def save_paths():
    path1 = entry_path1.get()
    path2 = entry_path2.get()
    path3 = entry_path3.get()
    
   
    with open("c:/Users/User/Desktop/paths.txt", "w") as file:
        file.write(f"Path 1: {path1}\n")
        file.write(f"Path 2: {path2}\n")
        file.write(f"Path 3: {path3}\n")
    
    
    label_message.config(text="Paths saved successfully!")


window = tk.Tk()


label_path1 = tk.Label(window, text="Path 1:")
label_path1.pack()
entry_path1 = tk.Entry(window)
entry_path1.pack()

label_path2 = tk.Label(window, text="Path 2:")
label_path2.pack()
entry_path2 = tk.Entry(window)
entry_path2.pack()

label_path3 = tk.Label(window, text="Path 3:")
label_path3.pack()
entry_path3 = tk.Entry(window)
entry_path3.pack()


button_save = tk.Button(window, text="Save Paths", command=save_paths)
button_save.pack()


label_message = tk.Label(window, text="")
label_message.pack()


window.mainloop()
