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

    # value = return_val[0]
    # movelist = return_val[1]
    # movetree = return_val[2]

    # print(movelist)

    return value, movelist, movetree


    raise NotImplementedError("you need to write this!")

def minimax_recursor(side, board, flags, depth, knight):
    if depth == 0:
        return evaluate(board), [], {}
    
    if depth > 0:
        comparator = {}
        movetree = {}
        
        for move in generateMoves(side, board, flags):
            # print(move)
            if move[0] == [2,1]:
                knight = 1
            else:
                knight = 0
            encoded_move = encode(move[0], move[1], move[2])
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            value, movelist, movetree_component = minimax_recursor(newside, newboard, newflags, depth - 1, knight)
            # print(movelist)
            if movelist == []:
                comparator[value] = [move]
            else:
                movelist.insert(0, move)
                # move.append(movelist)
                if depth == 2:
                    # if move[0][0] == [2,1]:
                    #     print(value, move)
                    if value == 5:
                        print(value, movelist)
                if value not in comparator:
                    comparator[value] = movelist
            # print("movelist")
            # print(comparator[value])
            # print("movelist")

            # if move[0] == [2,1]:
            #     print(comparator)
            movetree[encoded_move] = movetree_component
        chosen_key = 0
        if not side:
            chosen_key = max(comparator)
        else:
            chosen_key = min(comparator)
        if depth == 2:
            print(comparator)
            print(chosen_key)
        # if knight == 1:
            # print(comparator)
            # print(chosen_key)

        return chosen_key, comparator[chosen_key], movetree




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
    value, movelist, movetree, alpha, beta = alphabeta_recursor(side, board, flags, depth, alpha, beta)

    print(movelist)

    return value, movelist, movetree
    raise NotImplementedError("you need to write this!")
    

def alphabeta_recursor(side, board, flags, depth, alpha, beta):
    if depth == 0:
        return evaluate(board), [], {}, alpha, beta
    
    if depth > 0:
        movetree = {}
        movelist = []
        value = 0
        if side:
            value = math.inf
        else:
            value = -math.inf
        for move in generateMoves(side, board, flags):
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            value_return, temp_movelist, movetree_part, alpha, beta = alphabeta_recursor(newside, newboard, newflags, depth -1, alpha, beta)
            encoded_move = encode(move[0], move[1], move[2])
            movetree[encoded_move] = movetree_part


            if side:
                if value > value_return:
                    value = value_return
                    if movelist == []:
                        movelist = [move]
                    else:
                        print("checks")
                        temp_movelist.insert(0, move)
                        movelist = temp_movelist
                        print(movelist)
                if beta > value_return:
                    beta = value_return
                else:
                    break
            else:
                if value < value_return:
                    value = value_return
                    if movelist == []:
                        movelist = [move]
                    else:
                        print("checks")
                        temp_movelist.insert(0, move)
                        movelist = temp_movelist
                        print(movelist)
                if alpha < value_return:
                    alpha = value_return
                else:
                    break
        print(movelist)
        return value, movelist, movetree, alpha, beta
            
            


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
    raise NotImplementedError("you need to write this!")
