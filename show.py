import tkinter as tk
from tkinter import ttk, messagebox
import sys

class GoBoardUI:
    def __init__(self, root, is_boss_mode=False):
        self.root = root
        self.board_size = None  # No default size; user must set it
        self.cell_size = 30     # Each cell's size in pixels
        self.board = None       # Board initialized after size is set
        self.current_step = 0   # Current step
        self.moves = []         # Move list: [(pos, attr), ...]
        self.is_boss_mode = is_boss_mode
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(padx=10, pady=10)
        
        # Board size selection UI
        self.board_size_var = tk.StringVar()
        ttk.Label(self.main_frame, text="棋盘大小 n:").grid(row=0, column=0, padx=5)
        self.board_size_entry = ttk.Entry(self.main_frame, textvariable=self.board_size_var, width=5)
        self.board_size_entry.grid(row=0, column=1, padx=5)
        self.set_size_button = ttk.Button(self.main_frame, text="设置大小", command=self.set_board_size)
        self.set_size_button.grid(row=0, column=2, padx=5)
        self.current_size_label = ttk.Label(self.main_frame, text="当前: 未设置")
        self.current_size_label.grid(row=0, column=3, padx=5)
        
        # Input for move list
        self.input_label = ttk.Label(self.main_frame, text="输入移动列表 (例如 342, 0(full),333):")
        self.input_label.grid(row=1, column=0, columnspan=5, pady=5)
        
        self.input_text = tk.Text(self.main_frame, height=3, width=50)
        self.input_text.grid(row=2, column=0, columnspan=5, pady=5)
        
        self.load_button = ttk.Button(self.main_frame, text="加载", command=self.load_moves)
        self.load_button.grid(row=3, column=2, pady=5)
        
        # Canvas will be created after size is set
        self.canvas = None
        
        # Control components will be created after size is set
        self.slider = None
        self.prev_button = None
        self.next_button = None
        self.step_label = None
        self.info_label = None

    def set_board_size(self):
        """Set the board size based on user input and initialize the board."""
        try:
            n = int(self.board_size_var.get())
            if n <= 0:
                raise ValueError("n 必须是正整数")
            
            self.board_size = n
            self.board = [[0] * n for _ in range(n)]
            self.moves = []
            self.current_step = 0
            self.current_size_label.config(text=f"当前: {n} x {n}")
            
            # If canvas exists, reconfigure it; otherwise, create it
            if self.canvas:
                self.canvas.config(width=n * self.cell_size + 50, height=n * self.cell_size + 50)
                self.canvas.delete("all")
            else:
                self.canvas = tk.Canvas(
                    self.main_frame,
                    width=n * self.cell_size + 50,
                    height=n * self.cell_size + 50
                )
                self.canvas.grid(row=4, column=0, columnspan=5)
                
                # Initialize control components
                self.slider = ttk.Scale(
                    self.main_frame,
                    from_=0,
                    to=0,
                    orient=tk.HORIZONTAL,
                    length=200,
                    command=self.slider_moved
                )
                self.slider.grid(row=5, column=0, sticky='w', padx=(0, 5))
                
                self.prev_button = ttk.Button(self.main_frame, text="←", command=self.prev_move, width=5)
                self.prev_button.grid(row=5, column=1, padx=(0, 2))
                
                self.next_button = ttk.Button(self.main_frame, text="→", command=self.next_move, width=5)
                self.next_button.grid(row=5, column=2, padx=(0, 5))
                
                self.step_label = ttk.Label(self.main_frame, text="步数: 0")
                self.step_label.grid(row=5, column=3, padx=5)
                
                self.info_label = ttk.Label(self.main_frame, text="信息: ")
                self.info_label.grid(row=5, column=4, padx=5)
            
            self.draw_board()
            self.update_board(0)
        
        except ValueError as e:
            messagebox.showerror("错误", str(e) if str(e) != "n 必须是正整数" else "请输入有效的正整数")

    def draw_board(self):
        """Draw the grid lines for the board."""
        if not self.board_size or not self.canvas:
            return
        self.canvas.delete("all")
        margin = 25
        end = margin + (self.board_size - 1) * self.cell_size
        line_color = "white" if self.is_boss_mode else "black"
        for i in range(self.board_size):
            self.canvas.create_line(
                margin, margin + i * self.cell_size,
                end, margin + i * self.cell_size,
                fill=line_color
            )
            self.canvas.create_line(
                margin + i * self.cell_size, margin,
                margin + i * self.cell_size, end,
                fill=line_color
            )

    def pos_to_coords(self, pos):
        """Convert a 1D position to (x, y) coordinates."""
        if not self.board_size:
            return (0, 0)  # Should not occur after size is set
        x = pos % self.board_size
        y = pos // self.board_size
        return x, y

    def update_board(self, steps):
        """Update the board based on the current step."""
        if not self.board_size or not self.canvas:
            return
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.canvas.delete("stone", "number", "last_move")
        
        if self.is_boss_mode:
            for i in range(min(steps, len(self.moves))):
                pos, _ = self.moves[i]
                x, y = self.pos_to_coords(pos)
                canvas_x = 25 + x * self.cell_size
                canvas_y = 25 + y * self.cell_size
                number = str(i + 1)
                text_color = "black"  # Same color for simplicity
                self.canvas.create_text(
                    canvas_x, canvas_y,
                    text=number,
                    fill=text_color,
                    font=("Arial", 10),
                    tags="number"
                )
        else:
            for i in range(min(steps, len(self.moves))):
                pos, _ = self.moves[i]
                x, y = self.pos_to_coords(pos)
                color = 1 if i % 2 == 0 else 2
                self.board[y][x] = color
                canvas_x = 25 + x * self.cell_size
                canvas_y = 25 + y * self.cell_size
                self.canvas.create_oval(
                    canvas_x - 12, canvas_y - 12,
                    canvas_x + 12, canvas_y + 12,
                    fill="black" if color == 1 else "white",
                    tags="stone"
                )
        
        if steps > 0 and not self.is_boss_mode:
            last_pos, _ = self.moves[steps - 1]
            x, y = self.pos_to_coords(last_pos)
            canvas_x = 25 + x * self.cell_size
            canvas_y = 25 + y * self.cell_size
            self.canvas.create_oval(
                canvas_x - 6, canvas_y - 6,
                canvas_x + 6, canvas_y + 6,
                fill="red",
                tags="last_move"
            )
        
        if steps > 0:
            pos, attr = self.moves[steps - 1]
            info_text = f"{pos}({attr})" if attr else str(pos)
            self.info_label.config(text=f"信息: {info_text}")
        
        self.current_step = steps
        self.step_label.config(text=f"步数: {steps}")
        self.slider.set(steps)

    def next_move(self):
        """Go to the next move."""
        if self.current_step < len(self.moves):
            self.update_board(self.current_step + 1)

    def prev_move(self):
        """Go to the previous move."""
        if self.current_step > 0:
            self.update_board(self.current_step - 1)

    def slider_moved(self, value):
        """Update board when slider is moved."""
        steps = int(float(value))
        if steps != self.current_step:
            self.update_board(steps)

    def load_moves(self):
        """Load the move list from user input."""
        if not self.board_size:
            messagebox.showerror("错误", "请先设置棋盘大小")
            return
        input_str = self.input_text.get("1.0", tk.END).strip()
        try:
            moves_str = input_str.split(',')
            self.moves = []
            for move in moves_str:
                move = move.strip()
                if '(' in move and ')' in move:
                    pos_str, attr = move.split('(', 1)
                    pos = int(pos_str.strip())
                    attr = attr.rstrip(')').strip()
                    self.moves.append((pos, attr))
                else:
                    pos = int(move)
                    self.moves.append((pos, None))
            if not self.moves:
                raise ValueError("列表为空")
            self.slider.config(to=len(self.moves))
            self.update_board(0)
        except ValueError as e:
            messagebox.showerror("错误", f"输入格式错误: {str(e)}")

def main():
    is_boss_mode = 'boss' in sys.argv
    title = "data" if is_boss_mode else "棋盘"
    root = tk.Tk()
    root.title(title)
    app = GoBoardUI(root, is_boss_mode)
    root.mainloop()

if __name__ == "__main__":
    main()