import customtkinter as ctk
import sys
import os

# Add project root to sys.path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import screen modules
from ui.screens.home_screen import HomeScreen
from ui.screens.chat_screen import ChatScreen
from ui.screens.train_screen import TrainScreen
from ui.screens.update_screen import UpdateScreen


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("InfoPilot: AI 기반 문서 검색 엔진")
        self.geometry("1100x700")
        self.is_task_running = False

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Navigation Frame ---
        self.navigation_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        self.navigation_frame_label = ctk.CTkLabel(self.navigation_frame, text="InfoPilot",
                                                   font=ctk.CTkFont(size=20, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = ctk.CTkButton(self.navigation_frame, text="Home", command=lambda: self.select_frame("home"))
        self.home_button.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        self.chat_button = ctk.CTkButton(self.navigation_frame, text="Chat", command=lambda: self.select_frame("chat"))
        self.chat_button.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        self.train_button = ctk.CTkButton(self.navigation_frame, text="Train",
                                          command=lambda: self.select_frame("train"))
        self.train_button.grid(row=3, column=0, sticky="ew", padx=10, pady=5)

        self.update_button = ctk.CTkButton(self.navigation_frame, text="Update",
                                           command=lambda: self.select_frame("update"))
        self.update_button.grid(row=4, column=0, sticky="ew", padx=10, pady=5)

        # --- Main Content Frames ---
        # Pass the main app instance (self) to each screen
        self.home_frame = HomeScreen(self, corner_radius=0, fg_color="transparent")
        # Pass start_task and end_task methods as callbacks
        self.chat_frame = ChatScreen(self, corner_radius=0, fg_color="transparent")
        self.train_frame = TrainScreen(self, start_task_callback=self.start_task, end_task_callback=self.end_task,
                                       corner_radius=0, fg_color="transparent")
        self.update_frame = UpdateScreen(self, start_task_callback=self.start_task, end_task_callback=self.end_task,
                                         corner_radius=0, fg_color="transparent")

        self.select_frame("home")

    def select_frame(self, name):
        if self.is_task_running:
            print("Task is running, navigation is disabled.")
            return

        self.home_frame.grid_forget()
        self.chat_frame.grid_forget()
        self.train_frame.grid_forget()
        self.update_frame.grid_forget()

        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
            self.home_frame.refresh_state()
        elif name == "chat":
            self.chat_frame.grid(row=0, column=1, sticky="nsew")
            self.chat_frame.on_show()  # Call a method to re-initialize if needed
        elif name == "train":
            self.train_frame.grid(row=0, column=1, sticky="nsew")
        elif name == "update":
            self.update_frame.grid(row=0, column=1, sticky="nsew")
            self.update_frame.on_show()  # Call a method to refresh state

    def start_task(self):
        """Disables navigation when a long task starts."""
        self.is_task_running = True
        self.home_button.configure(state="disabled")
        self.chat_button.configure(state="disabled")
        self.train_button.configure(state="disabled")
        self.update_button.configure(state="disabled")

    def end_task(self):
        """Enables navigation when a long task ends."""
        self.is_task_running = False
        self.home_button.configure(state="normal")
        self.chat_button.configure(state="normal")
        self.train_button.configure(state="normal")
        self.update_button.configure(state="normal")


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = App()
    app.mainloop()
