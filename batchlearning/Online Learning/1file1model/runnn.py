import ast
import re

best_score = float('inf')
best_params = None

with open('model_scores.txt', 'r') as f:
    for line in f:
        # Extract the parameters and score from the line
        params_match = re.search(r"Params: ({.*}), Score:", line)
        score_match = re.search(r"Score: (.*)", line)

        if params_match is not None and score_match is not None:
            params_str = params_match.group(1)
            score_str = score_match.group(1)

            # Convert the parameters and score to Python objects
            params = ast.literal_eval(params_str)
            score = float(score_str)

            # Update the best model if this model has a lower score
            if score < best_score:
                best_score = score
                best_params = params
        else:
            print("No match found in line")

print(f"Best Score: {best_score}")
print(f"Best Params: {best_params}")