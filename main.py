import numpy as np

operations = []
steps = []  # save the correct steps done to achieve the required result in a stack


# Class to hold information regarding each step in the way to achieve the required result
class SolutionState:
    action = ""
    comment = ""
    top_left_corner_x = -1
    top_left_corner_y = -1
    name = ""
    front = True
    vertical = True
    rotate_deg = 0
    boardState = np.zeros([9, 9], dtype=float)
    current_piece = np.zeros([9, 3], dtype=float)
    original_piece = np.zeros([9, 3], dtype=float)

    def __init__(self, x, y, name, front, vertical, rotate_deg, boardState, action, current_piece, original_piece):
        self.top_left_corner_x = x
        self.top_left_corner_y = y
        self.name = name
        self.front = front
        self.vertical = vertical
        self.rotate_deg = rotate_deg
        self.boardState = boardState
        self.action = action
        self.comment = self.action + " " + name
        self.current_piece = current_piece
        self.original_piece = original_piece


def slice_board(x, y, r, c, arr):
    return arr[x: r + x, y: c + y]


def is_game_solved(arr):
    """
    Check whether the board is filled with ones (expect from the middle cell which should be empty, i.e. zero)
    """
    if arr[4][4] != 0:
        return False

    for r in range(9):
        for c in range(9):
            if arr[r][c] != 1:
                if not (r == 4 and c == 4):
                    return False

    return True


def search_for_possible_pos_to_place_piece(arr):
    """
    We try to place the current piece based on its top left corner.
    Therefore, the left column and the top row are the only optional cell to place each piece
    The odd rows and columns (1,3,5,7) can not be used for toop left corner as they are skeleton of the piece (the
    middle column). Therefore, the for loop done in jumps of 2 cells, checking only the top row and left column

    To make sure the cell is ready for placing piece in it, we check the next col/ row next to it (which indicate the
    skeleton), and make sure it's empty. if it's not empty it means that a piece is already placed at this location so
    we have no reason to check it

    The method return the first cell that is ready for insert check
    """
    col = 0
    for row in range(0, 7, 2):
        if arr[row + 1][col] != 1:
            return [row, col]

    row = 0
    for col in range(0, 7, 2):
        if arr[row][col + 1] != 1:
            return [row, col]

    return [-1, -1]



def solve_game(arr, all_p):
    """
    The main function whiach is responsible for two main things:
    1. Test for safe action
    2. Backtracking - I chose to use backtracking to solve this game.
        The game can be solved by one by one assigning pieces to possible sub-matrix in the board. Before assigning a
        piece, I check whether it is safe to assign - the piece is not used (check it before try other option of the
        same piece as flipping and rotating), the piece is not blocked by other pieces (this is done by validating the
        value of each cell in the board after summing up with the current piece).
        If the safety check is ok, I assign the piece to the board, and recursively check whether this assignment
        leads to a solution or not (using the stop condition to check if the game is solved)

        If the assignment didn’t lead to a solution, then I try the next piece / same piece in another configuration
        for the current checked top left cell
    :param arr: the game board
    :param all_p: a dictionary of all pieces
    :return:
    """
    if is_game_solved(arr):
        return True

    [row, col] = search_for_possible_pos_to_place_piece(arr)

    # consider pieces 1 to 8
    for num in range(1, len(all_p) + 1):
        # Get the current piece
        p = all_p[num]
        # If the piece is already used we should skip it
        if p.used:
            continue
        rotate_count = 2

        # If the tested top left cell is in the top row, the piece must be in vertical state.
        # likewise, if the tested top left cell is in the left column, the piece must be in horizontal state.
        # In case the top left cell is in both (0,0) - two states (horizontal & vertical) are optional
        if col != 0 and row == 0 and not p.vertical:
            p.rotate()
        if col == 0 and row != 0 and p.vertical:
            p.rotate()
        if row == 0 and col == 0:
            rotate_count = 4

        # Test all possible rotation states of the piece
        for rotate_idx in range(0, rotate_count):
            if rotate_count == 2:
                p.rotate()
            p.rotate()
            # Test the flipped option as well
            for flip_idx in range(0, 2):
                p.flip()
                # Check if action is safe
                if p.is_safe(row, col, arr):
                    # Assign to board
                    p.add_on_board(row, col, arr)
                    # recursively check whether this assignment leads to a solution or not
                    if solve_game(arr, all_p):
                        return True
                    else:
                        # If not, undo last action and update the steps stack accordingly
                        operations.pop()
                        steps.pop()
                        p.remove_from_board(row, col, arr)
    # this triggers backtracking
    return False


class Piece:
    pos_r = -1
    pos_c = -1
    name = "P"
    front = True
    vertical = True
    current_state = np.zeros([9, 3], dtype=float)
    original_state = np.zeros([9, 3], dtype=float)
    used = False
    deg = 0

    def rotate(self):
        """
        rotate the piece and keep track of the rotation degree
        """
        self.vertical = not self.vertical
        self.current_state = np.rot90(self.current_state, k=1, axes=(0, 1))
        self.deg = (self.deg + 90) % 360

    def flip(self):
        """
        flip the piece and keep track of its state (flipped or not)
        When we flip a piece the skeleton of the piece is changed - to indicate the state of the piece
        the holes in front state are marked with the number 1.5, and when the piece is upside down they marked with
        -0.5
        This way we forcing the holes of each piece to be connected only to the bumps of another pieces
        """
        if self.front:
            self.current_state = np.where(self.current_state == 1.5, -0.5, self.current_state)
        else:
            self.current_state = np.where(self.current_state == -0.5, 1.5, self.current_state)

        self.front = not self.front
        self.current_state = np.fliplr(self.current_state)

    def slice_piece_from_board(self, x, y, arr):
        """
        :param x: x of top left corner
        :param y: y of top left corner
        :param arr: the board
        :return: sub matrix from board with the same shape of the piece.
                return empty if the action is not valid.
        """
        return slice_board(x, y, self.current_state.shape[0], self.current_state.shape[1], arr)

    def is_safe(self, x, y, arr):
        """
        Check for a safe action - can we place current piece at the x,y ?
        """
        if self.used:
            return False
        answer = self.slice_piece_from_board(x, y, arr)
        # If the shape of sliced matrix from the board doesn't equal to the piece's shape it means this is not a valid
        # place to assign the piece as it has not enough space
        if answer.shape != self.current_state.shape:
            return False
        for i in range(self.current_state.shape[0]):
            for j in range(self.current_state.shape[1]):
                z = self.current_state[i][j] + answer[i][j]
                # the only valid value for cell are : -0.5, 0, 1, 1.5
                if z != -0.5 and z != 0 and z != 1 and z != 1.5:
                    return False

        return True

    def add_on_board(self, x, y, arr):
        self.pos_r = x
        self.pos_c = y
        operations.append(self.name)
        self.used = True
        for i in range(self.current_state.shape[0]):
            for j in range(self.current_state.shape[1]):
                arr[x + i][y + j] = arr[x + i][y + j] + self.current_state[i][j]

        # update the stated for final print
        current_state = SolutionState(x, y, self.name, self.front, self.vertical, self.deg, np.copy(arr), "Add",
                                      np.copy(self.current_state), np.copy(self.original_state))
        steps.append(current_state)

    def remove_from_board(self, x, y, arr):
        """
        remove piece from board by subtracting the piece values
        """
        for i in range(self.current_state.shape[0]):
            for j in range(self.current_state.shape[1]):
                arr[x + i][y + j] = arr[x + i][y + j] - self.current_state[i][j]
        self.used = False
        self.pos_r = -1
        self.pos_c = -1



def test_safe():
    x2 = piecesDict[2]
    x4 = piecesDict[4]
    x6 = piecesDict[6]
    print("------validation????-------")
    x2.rotate()
    if x2.is_safe(6, 0):
        x2.add_on_board(6, 0)
        print(board)

    print("------validation!-----------")
    x4.flip()
    # x4.rotate()
    x4.rotate()
    x4.rotate()
    print(x4.current_state)
    if x4.is_safe(0, 4):
        x4.add_on_board(0, 4)
        print(board)
    else:
        print("error")

    print("------validation?-----------")
    x6.flip()
    # x4.rotate()
    # x4.rotate()
    # x4.rotate()
    print(x6.current_state)
    if x6.is_safe(0, 0):
        x6.add_on_board(0, 0)
        print(board)
    else:
        print("error")


def UT_2_p(arr):
    """
    Test I used to test my logic at first, by making the problem very small it helped me find out a bug immediately
    :param arr:
    :return:
    """
    for i in range(len(arr)):
        for j in range(3, len(arr[i])):
            arr[i][j] = 1
    print(board)
    x1 = Piece()
    x1.name = "x1"
    x1.current_state = np.array([[0, 0, 1],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [1, 1, 1],
                                 ])
    x1_comp = Piece()
    x1_comp.current_state = np.ones((9, 3))
    sub = np.subtract(x1_comp.current_state, x1.current_state)
    x2 = Piece()
    x2.current_state = sub
    x2.rotate()
    x2.rotate()
    x2.front = False
    x2.name = "x2"
    dic = {1: x1,
           2: x2,
           }
    print("x1.state")
    print(x1.current_state)
    print("x2.state")
    print(x2.current_state)
    # print(np.add(x1.state, x2.state))
    # x2.rotate()
    # x2.rotate()
    # print(np.add(x1.state, x2.state))
    # exit(1)
    return dic


def define_game_pieces():
    """
    Method to define the pieces
    :return:
    """
    x1 = Piece()
    x2 = Piece()
    x3 = Piece()
    x4 = Piece()
    x5 = Piece()
    x6 = Piece()
    x7 = Piece()
    x8 = Piece()
    x1.name = "x1"
    x2.name = "x2"
    x3.name = "x3"
    x4.name = "x4"
    x5.name = "x5"
    x6.name = "x6"
    x7.name = "x7"
    x8.name = "x8"
    x1.current_state = np.array([[0, 1, 1],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [1, 1, 1],
                                 ])
    x2.current_state = np.array([[1, 1, 0],
                                 [0, 1.5, 0],
                                 [1, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 1],
                                 ])
    x3.current_state = np.array([[0, 1, 0],
                                 [0, 1.5, 0],
                                 [1, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [1, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 1],
                                 ])
    x4.current_state = np.array([[0, 1, 0],
                                 [0, 1.5, 0],
                                 [1, 1, 1],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 1],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 ])
    x5.current_state = np.array([[1, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 1],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [1, 1, 0],
                                 ])
    x6.current_state = np.array([[0, 1, 1],
                                 [0, 1.5, 0],
                                 [1, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 1],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 ])
    x7.current_state = np.array([[0, 1, 1],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 1],
                                 [0, 1.5, 0],
                                 [1, 1, 0],
                                 ])
    x8.current_state = np.array([[0, 1, 1],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [1, 1, 0],
                                 [0, 1.5, 0],
                                 [0, 1, 0],
                                 [0, 1.5, 0],
                                 [1, 1, 0],
                                 ])
    dic = {1: x1,
           2: x2,
           3: x3,
           4: x4,
           5: x5,
           6: x6,
           7: x7,
           8: x8,
           }

    # keep the original state for final print
    for x in dic:
        dic[x].original_state = dic[x].current_state

    return dic


def printSteps():
    reverse_steps = []
    steps_count = len(steps)
    for i in range(steps_count):
        reverse_steps.append(steps.pop())

    for i in range(steps_count):
        print("~~~~~~~~~~~ step ", i + 1, " ~~~~~~~~~~~")
        p = reverse_steps.pop()
        isFlip = ""
        if p.front:
            isFlip = "is not flipped"
        else:
            isFlip = "is flipped"

        print(" Add ", p.name, " on board, at position ", p.top_left_corner_x, p.top_left_corner_y, ", rotated ",
              p.rotate_deg, " degrees and ", isFlip)

        print("Original ", p.name, ":")
        print(p.original_piece)

        print("Insert to board ", p.name, " as:")
        print(p.current_piece)

        print("Board after action:")
        print(p.boardState)




if __name__ == '__main__':

    # Init board
    board = np.zeros([9, 9], dtype=float)
    piecesDict = define_game_pieces()
    # piecesDict = UT_2_p(board)
    print("--------Board----------")
    print(board)
    if solve_game(board, piecesDict):
        print("Success")
        print(operations)
        print(board)
        print("~~~~~~~~~~~ steps ~~~~~~~")
        printSteps()
    else:
        print("No solution exists")

