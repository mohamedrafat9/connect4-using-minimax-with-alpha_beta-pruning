import tkinter as tk
from tkinter import messagebox, font
import numpy as np
import random
import math
import copy
import time

class Connect4:
    def __init__(self):
        self.ROW_COUNT, self.COLUMN_COUNT = 6, 7
        self.EMPTY = 0
        self.PLAYER, self.AI = 0, 1
        self.PLAYER_PIECE, self.AI_PIECE = 1, 2
        self.WINDOW_LENGTH = 4
        self.board = self.create_board()
        self.turn = self.AI 
        self.first_player = self.AI  
        self.game_over = False
        self.transposition = {}
        self.time_limit = 0.8

    def create_board(self):
        return np.zeros((self.ROW_COUNT, self.COLUMN_COUNT), dtype=int)

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def is_valid_location(self, col):
        return self.board[self.ROW_COUNT-1][col] == self.EMPTY

    def get_next_open_row(self, col):
        for r in range(self.ROW_COUNT):
            if self.board[r][col] == self.EMPTY:
                return r
        return -1

    def get_valid_locations(self):
        return [c for c in range(self.COLUMN_COUNT) if self.is_valid_location(c)]

    def winning_move(self, piece):
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT):
                if all(self.board[r][c+i] == piece for i in range(4)):
                    return True
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT-3):
                if all(self.board[r+i][c] == piece for i in range(4)):
                    return True
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT-3):
                if all(self.board[r+i][c+i] == piece for i in range(4)) or all(self.board[r+3-i][c+i] == piece for i in range(4)):
                    return True
        return False

    def is_terminal_node(self):
        return self.winning_move(self.PLAYER_PIECE) or self.winning_move(self.AI_PIECE) or not self.get_valid_locations()

    def score_position(self, piece):
        score = 0
        center = self.COLUMN_COUNT // 2
        score += list(self.board[:, center]).count(piece) * 6
        for offset, wt in [(-1,4),(1,4),(-2,2),(2,2)]:
            c = center + offset
            if 0 <= c < self.COLUMN_COUNT:
                score += list(self.board[:, c]).count(piece) * wt

        for r in range(self.ROW_COUNT):
            for c in range(self.COLUMN_COUNT-3):
                score += self.evaluate_window(list(self.board[r, c:c+4]), piece)

        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT-3):
                score += self.evaluate_window(list(self.board[r:r+4, c]), piece)

        for r in range(self.ROW_COUNT-3):
            for c in range(self.COLUMN_COUNT-3):
                score += self.evaluate_window([self.board[r+i][c+i] for i in range(4)], piece)
                score += self.evaluate_window([self.board[r+3-i][c+i] for i in range(4)], piece)
        return score

    def evaluate_window(self, window, piece):
        score = 0
        opp = self.PLAYER_PIECE if piece == self.AI_PIECE else self.AI_PIECE
        cnt_self, cnt_opp, cnt_empty = window.count(piece), window.count(opp), window.count(self.EMPTY)
        if cnt_self == 4:
            score += 1000
        elif cnt_self == 3 and cnt_empty == 1:
            score += 50
        elif cnt_self == 2 and cnt_empty == 2:
            score += 10
        if cnt_opp == 3 and cnt_empty == 1:
            score -= 80
        return score

    def move_score(self, col, piece):
        row = self.get_next_open_row(col)
        temp = copy.deepcopy(self)
        temp.drop_piece(row, col, piece)
        return temp.score_position(piece)

    def minimax(self, depth, alpha, beta, maximizingPlayer, start=None):
        if start and time.time() - start > self.time_limit:
            return None, self.score_position(self.AI_PIECE)
        key = (tuple(self.board.flatten()), depth, maximizingPlayer)
        if key in self.transposition:
            return self.transposition[key]
        valid = self.get_valid_locations()
        terminal = self.is_terminal_node()
        if depth == 0 or terminal:
            if terminal:
                if self.winning_move(self.AI_PIECE):
                    return (None, float('inf'))
                if self.winning_move(self.PLAYER_PIECE):
                    return (None, -float('inf'))
                return (None, 0)
            return (None, self.score_position(self.AI_PIECE))
        if maximizingPlayer:
            value, best = -math.inf, valid[0]
            valid.sort(key=lambda c: self.move_score(c, self.AI_PIECE), reverse=True)
            for col in valid:
                row = self.get_next_open_row(col)
                child = copy.deepcopy(self)
                child.drop_piece(row, col, self.AI_PIECE)
                _, sc = child.minimax(depth-1, alpha, beta, False, start)
                if sc is None:
                    break
                if sc > value:
                    value, best = sc, col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            res = (best, value)
        else:
            value, best = math.inf, valid[0]
            valid.sort(key=lambda c: self.move_score(c, self.PLAYER_PIECE))
            for col in valid:
                row = self.get_next_open_row(col)
                child = copy.deepcopy(self)
                child.drop_piece(row, col, self.PLAYER_PIECE)
                _, sc = child.minimax(depth-1, alpha, beta, True, start)
                if sc is None:
                    break
                if sc < value:
                    value, best = sc, col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            res = (best, value)
        self.transposition[key] = res
        return res

    def player_move(self, col):
        if not self.is_valid_location(col):
            return False
        r = self.get_next_open_row(col)
        self.drop_piece(r, col, self.PLAYER_PIECE)
        if self.winning_move(self.PLAYER_PIECE):
            self.game_over = True
            return "player_win"
        if not self.get_valid_locations():
            self.game_over = True
            return "draw"
        self.turn = self.AI
        return "continue"

    def bot_move(self):
        for piece in (self.AI_PIECE, self.PLAYER_PIECE):
            for col in self.get_valid_locations():
                r = self.get_next_open_row(col)
                temp = copy.deepcopy(self)
                temp.drop_piece(r, col, piece)
                if temp.winning_move(piece):
                    move = col
                    break
            else:
                continue
            break
        else:
            move, _ = self.minimax(6, -math.inf, math.inf, True, time.time())
        if move is not None and self.is_valid_location(move):
            r = self.get_next_open_row(move)
            self.drop_piece(r, move, self.AI_PIECE)
            if self.winning_move(self.AI_PIECE):
                self.game_over = True
                return "bot_win"
            if not self.get_valid_locations():
                self.game_over = True
                return "draw"
            self.turn = self.PLAYER
            return "continue"
        return None

    def reset_game(self):
        self.board = self.create_board()
        self.game_over = False
        self.turn = self.first_player  
        self.transposition.clear()

class Connect4GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Connect 4")
        self.master.geometry("800x700")
        self.master.resizable(False, False)
        self.master.configure(bg="#1E1E1E")
        self.game = Connect4()
        self.circle_size = 80
        self.padding = 20
        self.colors = {0: "black", 1: "#e74856", 2: "#fbcb41"}
        self.main_frame = tk.Frame(self.master, bg="black")
        self.game_frame = tk.Frame(self.master, bg="#1E1E1E")
        self.first_player_var = tk.StringVar(value="AI")  # Default AI starts
        self.main_menu()

    def main_menu(self):
        self.clear_frame()
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        title_font = font.Font(family="Helvetica", size=32, weight="bold")
        title = tk.Label(self.main_frame, text="Welcome to our Connect 4 Game", fg="#ffffff", bg="black", font=title_font)
        title.pack(pady=(100, 20))

        # First player selection
        select_frame = tk.Frame(self.main_frame, bg="black")
        select_frame.pack(pady=20)
        select_label = tk.Label(select_frame, text="Who plays first?", fg="#ffffff", bg="black", font=("Helvetica", 16))
        select_label.pack()
        tk.Radiobutton(select_frame, text="Player (Red)", variable=self.first_player_var, value="Player",
                       fg="#ffffff", bg="black", selectcolor="black", font=("Helvetica", 14)).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(select_frame, text="AI (Yellow)", variable=self.first_player_var, value="AI",
                       fg="#ffffff", bg="black", selectcolor="black", font=("Helvetica", 14)).pack(side=tk.LEFT, padx=10)

        btn_font = font.Font(family="Helvetica", size=16)
        play_btn = tk.Button(self.main_frame, text="Player vs AI", command=lambda: self.start_game("ai"),
                             fg="#ffffff", bg="#007ACC", activebackground="#3399FF", font=btn_font,
                             width=20, height=2, bd=5)
        play_btn.pack(pady=20)
        exit_btn = tk.Button(self.main_frame, text="Exit", command=self.master.quit,
                             fg="#ffffff", bg="#E81123", activebackground="#FF3F3F", font=btn_font,
                             width=10, height=1, bd=2)
        exit_btn.pack(pady=(20, 50))

    def start_game(self, mode):
        self.game.first_player = self.game.PLAYER if self.first_player_var.get() == "Player" else self.game.AI
        self.game.reset_game()
        self.game.player_mode = mode
        self.clear_frame()
        self.game_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas_width = self.game.COLUMN_COUNT * (self.circle_size + self.padding) + self.padding
        self.canvas_height = self.game.ROW_COUNT * (self.circle_size + self.padding) + self.padding
        self.canvas = tk.Canvas(self.game_frame, width=self.canvas_width, height=self.canvas_height,
                                bg="black", bd=0, highlightthickness=0)
        self.canvas.pack(pady=20)
        self.canvas.bind("<Button-1>", self.handle_click)
        self.create_control_buttons()
        self.draw_board()
        if mode == "ai" and self.game.turn == self.game.AI:
            self.master.after(200, self.ai_move)

    def create_control_buttons(self):
        frame = tk.Frame(self.game_frame, bg="#1E1E1E")
        frame.pack(pady=10)
        btn_font = font.Font(family="Helvetica", size=12)
        menu_btn = tk.Button(frame, text="Main Menu", command=self.main_menu,
                             fg="#ffffff", bg="#007ACC", activebackground="#3399FF", font=btn_font,
                             width=12, bd=0)
        menu_btn.pack(side=tk.LEFT, padx=5)
        reset_btn = tk.Button(frame, text="Reset Game", command=self.reset_game,
                              fg="#ffffff", bg="#107C10", activebackground="#28A428", font=btn_font,
                              width=12, bd=0)
        reset_btn.pack(side=tk.LEFT, padx=5)
        exit_btn = tk.Button(frame, text="Exit", command=self.master.quit,
                             fg="#ffffff", bg="#E81123", activebackground="#FF3F3F", font=btn_font,
                             width=12, bd=0)
        exit_btn.pack(side=tk.LEFT, padx=5)

    def draw_board(self):
        self.canvas.delete("all")
        for c in range(self.game.COLUMN_COUNT):
            for r in range(self.game.ROW_COUNT):
                x1 = c * (self.circle_size + self.padding) + self.padding
                y1 = (self.game.ROW_COUNT - r - 1) * (self.circle_size + self.padding) + self.padding
                x2, y2 = x1 + self.circle_size, y1 + self.circle_size
                self.canvas.create_oval(x1, y1, x2, y2,
                                        fill=self.colors[self.game.board[r][c]], outline="#C5C5C5", width=2)

    def handle_click(self, event):
        if self.game.game_over or (self.game.player_mode == "ai" and self.game.turn == self.game.AI):
            return
        col = event.x // (self.circle_size + self.padding)
        if 0 <= col < self.game.COLUMN_COUNT:
            result = self.game.player_move(col)
            self.draw_board()
            if result == "player_win":
                messagebox.showinfo("Game Over", "Player (Red) wins!")
                return
            elif result == "draw":
                messagebox.showinfo("Game Over", "It's a draw!")
                return
            if self.game.player_mode == "ai":
                self.master.after(200, self.ai_move)

    def ai_move(self):
        if self.game.game_over or self.game.turn != self.game.AI:
            return
        result = self.game.bot_move()
        self.draw_board()
        self.master.update_idletasks()
        if result == "bot_win":
            messagebox.showinfo("Game Over", "AI (Yellow) wins!")
        elif result == "draw":
            messagebox.showinfo("Game Over", "It's a draw!")

    def reset_game(self):
        self.game.first_player = self.game.PLAYER if self.first_player_var.get() == "Player" else self.game.AI
        self.game.reset_game()
        self.draw_board()
        if self.game.player_mode == "ai" and self.game.turn == self.game.AI:
            self.master.after(200, self.ai_move)

    def clear_frame(self):
        for f in (self.main_frame, self.game_frame):
            f.pack_forget()

if __name__ == "__main__":
    root = tk.Tk()
    app = Connect4GUI(root)
    root.mainloop()
