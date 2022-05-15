from cmath import inf
from distutils.file_util import move_file
import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    value, movelist, movetree = minimax_recursor(side, board, flags, depth, 0)


    return value, movelist, movetree


    raise NotImplementedError("you need to write this!")

def minimax_recursor(side, board, flags, depth, knight):
    if depth == 0:
        return evaluate(board), [], {}

    if depth > 0:
        best_val = 0
        if side:
            best_val = inf
        else:
            best_val = -inf
        best_move_list = []
        movetree = {}
        for move in generateMoves(side, board, flags):

            encoded_move = encode(move[0], move[1], move[2])
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            value, movelist, movetree_component = minimax_recursor(newside, newboard, newflags, depth - 1, knight)
            movelist.insert(0,move)
            if side:
                if value < best_val:
                    best_val = value
                    best_move_list = movelist
            else:
                if value > best_val:
                    best_val = value
                    best_move_list = movelist


            movetree[encoded_move] = movetree_component
        

        return best_val, best_move_list, movetree




def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    print(depth)
    value, movelist, movetree = alphabeta_recursor_min(side, board, flags, depth, alpha, beta)

    print(movelist)

    return value, movelist, movetree
    raise NotImplementedError("you need to write this!")




def alphabeta_recursor_max(side, board, flags, depth, alpha, beta):
    if depth == 0:
        return evaluate(board), [], {}
    if depth > 0:
        movetree = {}
        best_move_list = []
        for move in generateMoves(side, board, flags):
            encoded_move = encode(move[0], move[1], move[2])
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            value, movelist, movetree_component = alphabeta_recursor_min(newside, newboard, newflags, depth-1, alpha, beta)
            movetree[encoded_move] = movetree_component
            movelist.insert(0,move)
            if value >= beta:
                return beta, movelist, movetree
            if value > alpha:
                alpha = value
                best_move_list = movelist

        return alpha, best_move_list, movetree




def alphabeta_recursor_min(side, board, flags, depth, alpha, beta):
    if depth == 0:
        return evaluate(board), [], {}
    if depth > 0:
        movetree = {}
        best_move_list = []
        for move in generateMoves(side, board, flags):
            encoded_move = encode(move[0], move[1], move[2])
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            value, movelist, movetree_component = alphabeta_recursor_max(newside, newboard, newflags, depth-1, alpha, beta)
            movetree[encoded_move] = movetree_component
            movelist.insert(0,move)
            if value <= alpha:
                return alpha, movelist, movetree
            if value < beta:
                beta = value
                best_move_list = movelist

        return beta, best_move_list, movetree




def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    movetree = {}
    best_movelist = []
    best_val = 0
    if side:
        best_val = inf
    else:
        best_val = -inf
    for move in generateMoves(side, board, flags):
        encoded_move = encode(move[0], move[1], move[2])
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value, movelist, movetree_component = stochastic_recurser(newside, newboard, newflags, depth-1, breadth, chooser)
        movelist.insert(0,move)
        movetree[encoded_move] = movetree_component
        if side:
            if value < best_val:
                best_val = value
                best_movelist = movelist
        else:
            if best_val < value:
                best_val = value
                best_movelist = movelist

    return best_val, best_movelist, movetree


    raise NotImplementedError("you need to write this!")

def stochastic_recurser(side, board, flags, depth, breadth, chooser):
    if depth == 0 or breadth == 0:
        return evaluate(board), [], {}
    
    if depth > 0:
        movetree = {}
        best_movelist = []
        avg_val = 0
        
        moves = [ move for move in generateMoves(side, board, flags) ]
        for i in range(breadth):
            move = chooser(moves)
            encoded_move = encode(move[0], move[1], move[2])
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            value, movelist, movetree_component = stochastic_recurser(newside, newboard, newflags, depth-1, breadth, chooser)
            avg_val += value
            movetree[encoded_move] = movetree_component
            movelist.insert(0,move)
            best_movelist = movelist
        avg_val /= breadth
        return avg_val, best_movelist, movetree