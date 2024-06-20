import random

import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(maze, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next = (current[0] + dx, current[1] + dy)
            if maze[next[0]][next[1]] != 'W':
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

    return came_from, cost_so_far

def draw_path(maze, came_from, start, goal):
    current = goal
    while current != start:
        maze[current[0]][current[1]] = 'P'
        current = came_from[current]

def generate_maze(width: int, height: int) -> list:
    # Create the outline of the maze with walls
    maze = [['W' for _ in range(width)] for _ in range(height)]
    
    # List of possible positions for paths
    path_positions = [(i, j) for i in range(1, height-1, 2) for j in range(1, width-1, 2)]
    
    # Start Recursive Backtracker from a random position
    start_pos = random.choice(path_positions)
    maze[start_pos[0]][start_pos[1]] = 'N'
    path_positions.remove(start_pos)
    stack = [start_pos]
    
    while stack:
        current_pos = stack[-1]
        
        # Get all neighboring cells
        neighbors = [(current_pos[0] + dx * 2, current_pos[1] + dy * 2) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        valid_neighbors = [pos for pos in neighbors if pos in path_positions]
        
        if valid_neighbors:
            next_pos = random.choice(valid_neighbors)
            maze[next_pos[0]][next_pos[1]] = 'N'
            path_positions.remove(next_pos)
            stack.append(next_pos)
            
            # Remove the wall between the current cell and the chosen cell
            maze[current_pos[0] + (next_pos[0] - current_pos[0]) // 2][current_pos[1] + (next_pos[1] - current_pos[1]) // 2] = 'N'
        else:
            stack.pop()  # Backtrack if no valid neighbors

    # List of possible positions for player, targets and start
    empty_positions = [(i, j) for i in range(1, height-1) for j in range(1, width-1) if maze[i][j] == 'N']
    
    # Randomly select positions for player, target A, target B and start
    p_pos, a_pos, b_pos, s_pos = random.sample(empty_positions, 4)
    maze[p_pos[0]][p_pos[1]] = 'P'
    maze[a_pos[0]][a_pos[1]] = 'A'
    maze[b_pos[0]][b_pos[1]] = 'B'
    maze[s_pos[0]][s_pos[1]] = 'S'
    
    return maze, p_pos, a_pos, b_pos, s_pos

import tkinter as tk

def draw_maze(maze):
    root = tk.Tk()
    rows, cols = len(maze), len(maze[0])
    canvas = tk.Canvas(root, width=cols*20, height=rows*20, bg='white')
    canvas.pack()

    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            x1, y1 = j * 20, i * 20
            x2, y2 = x1 + 20, y1 + 20
            if cell == 'W':
                color = 'black'
            elif cell == 'P':
                color = 'blue'
            elif cell == 'A':
                color = 'red'
            elif cell == 'B':
                color = 'green'
            else:
                color = 'white'
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')

    root.mainloop()

maze, p_pos, a_pos, b_pos, s_pos = generate_maze(50, 50)
came_from, _ = a_star_search(maze, p_pos, a_pos)
draw_path(maze, came_from, p_pos, a_pos)
came_from, _ = a_star_search(maze, a_pos, b_pos)
draw_path(maze, came_from, a_pos, b_pos)
came_from, _ = a_star_search(maze, b_pos, s_pos)
draw_path(maze, came_from, b_pos, s_pos)
draw_maze(maze)