import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import subprocess
from train import *

STATE_LAYER_NUM = 3

# Define the adjustable board size
size = 20  # Change this value to adjust the board size (e.g., 15, 10, etc.)

# Five class to manage the game state
class Five:
    def __init__(self, size):
        """Initialize the game state with a given size"""
        self.size = size
        self.board = [0] * (size * size)  # 0 for empty, 1 for player 1, -1 for player 2
        self.current_player = 1  # Current player: 1 or -1
        self.all_moves = []

    def make_move(self, position):
        """Make a move at the specified position"""
        if self.board[position] != 0:
            return
        self.board[position] = self.current_player
        self.current_player *= -1
        self.all_moves.append(position)

# Function to get the game state from the C++ executable
def get_state(size, actions):
    """Get the game state by calling the C++ executable"""
    cmd = ['./trtangel_release', 'getstate', str(size)] + [str(a) for a in actions]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Error in getstate: " + result.stderr)
    state_str = result.stdout.strip()
    state = [float(x) for x in state_str.split()]
    return state

# Initialize game and model
game = Five(size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KataNet().to(device)
# Load the pre-trained model for the specified size

model.load_state_dict(torch.load(f'model/model.pth', map_location=device))
model.eval()

# Create UI
root = tk.Tk()
root.title("ui")

# Create a size x size board, canvas dimensions are 50 * size pixels
canvas = tk.Canvas(root, width=50 * size, height=50 * size)
canvas.pack()

# Draw board lines
for i in range(size):
    canvas.create_line(50 * i + 25, 25, 50 * i + 25, 50 * size - 25)  # Vertical
    canvas.create_line(25, 50 * i + 25, 50 * size - 25, 50 * i + 25)  # Horizontal

# Draw row and column labels
for i in range(size):
    canvas.create_text(10, 50 * i + 25, text=str(i), font=("Arial", 12), anchor="e")  # Row numbers
    canvas.create_text(50 * i + 25, 10, text=str(i), font=("Arial", 12), anchor="s")  # Column numbers

# Function to draw a piece on the board
def draw_piece(row, col, player):
    """Draw a piece at the specified row and column"""
    x = 50 * col + 25
    y = 50 * row + 25
    if player == 1:
        canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill="black")
    elif player == -1:
        canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill="white")

# Function to display the AI's policy
def show_policy(policy):
    """Display the AI's policy on the board"""
    canvas.delete("policy")  # Clear previous policy display
    available = [pos for pos in range(size * size) if game.board[pos] == 0]
    if not available:
        return

    # Get probabilities for available positions and sort them
    probs = [(pos, policy[pos]) for pos in available]
    probs.sort(key=lambda x: x[1], reverse=True)  # Sort by probability in descending order

    # Define ranking thresholds
    total = len(probs)
    top_10 = int(total * 0.1)  # Top 10%
    top_30 = int(total * 0.3)  # Top 30%
    top_70 = int(total * 0.7)  # Top 70%

    # Assign colors based on ranking
    rank_dict = {}
    for i, (pos, _) in enumerate(probs):
        if i < top_10:
            rank_dict[pos] = 'green'    # Top tier
        elif i < top_30:
            rank_dict[pos] = 'yellow'   # High tier
        elif i < top_70:
            rank_dict[pos] = 'red'      # Mid tier
        else:
            rank_dict[pos] = 'purple'   # Low tier

    # Highlight the position with the highest probability
    max_pos = probs[0][0]

    # Draw policy probabilities on the board
    for pos in available:
        row, col = divmod(pos, size)
        x = 50 * col + 25
        y = 50 * row + 25
        prob = policy[pos]
        color = rank_dict[pos]

        # Draw background rectangle
        canvas.create_rectangle(x - 20, y - 10, x + 20, y + 10, fill="lightgray", tags="policy")

        # Draw a blue outline for the best move
        if pos == max_pos:
            canvas.create_rectangle(x - 20, y - 20, x + 20, y + 20, outline="blue", width=2, tags="policy")

        # Draw probability text
        canvas.create_text(x, y, text=f"{prob:.3f}", fill=color, tags="policy", font=("Arial", 14))

# Function to get and display the AI's policy
def get_and_show_policy():
    """Get the game state and display the AI's policy"""
    state = torch.tensor(get_state(size, game.all_moves), dtype=torch.float32, device=device).view(1, STATE_LAYER_NUM, 20, 20).to(device)
    with torch.no_grad():
        board_sizes = torch.tensor([size], dtype=torch.float32, device=device)
        policy_logits, value_logits = model(state, board_sizes)

        policy_logits_current = policy_logits[0, 0, :, :size]  # [20, size]，只取有效区域
        policy_logits_local = policy_logits_current[:size, :].flatten()  # [size*size]
        policy = F.softmax(policy_logits_local, dim=0)  # (1, size * size) 计算策略概率，注意需要转换为局部坐标

        value_probs = F.softmax(value_logits, dim=1)  # 将 logits 转换为概率分布
        win_prob = value_probs[0, 0]  # win 的概率
        lose_prob = value_probs[0, 1]  # lose 的概率
        reward = win_prob - lose_prob

    policy = policy.cpu().numpy()
    print("发送state: ", game.all_moves, "黑方得分: ", reward if game.current_player == 1 else -reward)
    show_policy(policy)

# Function to handle user moves
def user_move(event):
    """Handle a user's click to place a piece"""
    x, y = event.x, event.y
    # Calculate the nearest grid intersection
    col = round((x - 25) / 50)
    row = round((y - 25) / 50)
    # Check if the position is within the board
    if 0 <= row < size and 0 <= col < size:
        pos = row * size + col
        if game.board[pos] == 0:
            current = game.current_player
            game.make_move(pos)
            print(pos)
            draw_piece(row, col, current)
            get_and_show_policy()  # Update AI policy display

# Bind mouse click event
canvas.bind("<Button-1>", user_move)

# Start the game
get_and_show_policy()  # Display initial policy
root.mainloop()
