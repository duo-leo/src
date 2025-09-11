import tkinter as tk
from tkinter import ttk

class ManualFixer:
    def __init__(self, phonetic_line, phonetic_options, current_page=1, total_pages=20):
        self.phonetic_line = phonetic_line
        if phonetic_options is None:
            phonetic_options = [[] for _ in range(len(phonetic_line))]
        self.phonetic_options = phonetic_options
        self.selected_options = [None] * len(phonetic_line)  # To save changes
        self.button_references = []  # To store references to the buttons
        self.current_page = current_page
        self.total_pages = total_pages

        self.root = tk.Tk()
        self.root.title("Phonetic Line Selector")
        self.root.geometry("1200x600")

        self.create_main_ui()

    def create_main_ui(self):
        # Frame for displaying page and progress
        header_frame = tk.Frame(self.root)
        header_frame.pack(pady=5, padx=10, fill=tk.X)

        # Page display
        self.page_label = tk.Label(header_frame, text=f"Page {self.current_page}")
        self.page_label.pack(side=tk.LEFT, padx=5)

        # Frame for phonetic_line buttons as a horizontal scrollable list
        scroll_frame = tk.Frame(self.root)
        scroll_frame.pack(pady=10, padx=10, fill=tk.X)

        canvas = tk.Canvas(scroll_frame, height=50)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="horizontal", command=canvas.xview)
        button_frame = tk.Frame(canvas)

        button_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=button_frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar.set)

        for i, phonetic in enumerate(self.phonetic_line):
            btn = tk.Button(
                button_frame, text=phonetic, command=lambda idx=i: self.show_options(idx)
            )
            btn.pack(side=tk.LEFT, padx=5)

            if len(self.phonetic_options[i]) == 1:
                btn.config(bg="limegreen")
            else:
                btn.config(bg="coral")

            self.button_references.append(btn)  # Save reference to the button

        canvas.pack(fill=tk.X, expand=True)
        scrollbar.pack(fill=tk.X)

        # Frame for phonetic_options
        self.options_frame = tk.Frame(self.root)
        self.options_frame.pack(pady=10, padx=10)

        # Whole line input field
        self.line_input_label = tk.Label(self.root, text="Edit Whole Line:")
        self.line_input_label.pack(pady=5)

        self.line_input_field = tk.Entry(self.root, width=50)
        self.line_input_field.pack(pady=5)
        self.line_input_field.insert(0, " ".join(self.phonetic_line))  # Display current line
        self.line_input_field.bind("<Return>", self.update_whole_line)  # Save on Enter

        # Save whole line button
        save_line_btn = tk.Button(self.root, text="Save Whole Line", command=self.update_whole_line)
        save_line_btn.pack(pady=5)

        # Finish button
        finish_btn = tk.Button(self.root, text="Finish", command=self.finish)
        finish_btn.pack(pady=10)

    def show_options(self, index):
        # Clear previous options
        for widget in self.options_frame.winfo_children():
            widget.destroy()

        options = self.phonetic_options[index]
        if len(options) > 1:
            tk.Label(self.options_frame, text=f"Options for {self.phonetic_line[index]}:").pack()

            # Horizontal scrollable list
            canvas = tk.Canvas(self.options_frame, height=50)
            scrollbar = ttk.Scrollbar(self.options_frame, orient="horizontal", command=canvas.xview)
            scrollable_frame = tk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(xscrollcommand=scrollbar.set)

            for opt in options:
                btn = tk.Button(
                    scrollable_frame,
                    text=opt,
                    command=lambda opt=opt: self.save_option(index, opt),
                )
                btn.pack(side=tk.LEFT, padx=5)

                if opt == self.selected_options[index]:
                    btn.config(bg="limegreen")

            canvas.pack(fill=tk.X, expand=True)
            scrollbar.pack(fill=tk.X)
        elif len(options) == 1:
            # Show text field directly if only one option
            self.save_option(index, options[0])
        else:
            tk.Label(self.options_frame, text="No options available.").pack()

        # Manual input field
        tk.Label(self.options_frame, text="Manual Input:").pack(pady=5)
        manual_entry = tk.Entry(self.options_frame)
        manual_entry.pack(pady=5)
        manual_entry.bind("<KeyRelease>", lambda e: self.save_option(index, manual_entry.get()))
        # manual_entry.bind("<Return>", lambda e: self.save_option(index, manual_entry.get()))

        # set text to the last saved option
        if self.selected_options[index] is not None:
            manual_entry.insert(0, self.selected_options[index])
        else:
            manual_entry.insert(0, self.phonetic_line[index])

    def save_option(self, index, option):
        self.selected_options[index] = option
        self.phonetic_line[index] = option
        print(f"Saved for {self.phonetic_line[index]}: {option}")  # Debugging

        # Update phonetic button color and text
        self.button_references[index].configure(bg="yellow", text=option)

        # Update whole line input field
        self.line_input_field.delete(0, tk.END)
        self.line_input_field.insert(0, " ".join(self.phonetic_line))

    def update_whole_line(self, event=None):
        # Update phonetic_line based on whole line input
        updated_line = self.line_input_field.get()
        self.phonetic_line = updated_line.split(" ")

        for i in range(len(self.phonetic_line)):
            self.button_references[i].configure(text= self.phonetic_line[i], bg="yellow")

        print("Whole line updated:", self.phonetic_line)

    def finish(self):
        print("Final selected options:", self.selected_options)
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.phonetic_line


# Input data
# phonetic_line = ["Line1", "Line2", "Line3", "Line4", "Line5", "Line6", "Line7", "Line8", "Line9"]
# phonetic_options = [
#     ["Option1", "Option2", "Option3", "Line1"],
#     ["OptionA", "OptionB", "Line2"],
#     ["OptionX", "OptionY", "OptionZ", "Line3"],
#     ["Line4"],
#     ["Option5", "Option6", "Option7", "Option8", "Line5"],
#     ["OptionM", "OptionN", "OptionO", "Line6"],
#     ["OptionP", "OptionQ", "OptionR", "Line7"],
#     ["Option9", "Line8"],
#     ["Option10", "Option11", "Line9"],
# ]
# # Launch the UI
# ui = ManualFixer(phonetic_line, phonetic_options)
# selected = ui.run()
# print("Final selected options:", selected)
