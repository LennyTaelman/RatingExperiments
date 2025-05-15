import jax
jax.config.update("jax_enable_x64", True) # Enable 64-bit precision from the start
import jax.numpy as jnp
from jax import lax
import numpy as np
import json
import argparse
from collections import defaultdict
from coordinate_newton_optimizer import coordinate_newton_optimize # Import the new optimizer

# Parameters for maximum likelihood estimation in the inference phase (priors)
DEFAULT_INF_PROB_DIST = {"mean": 15.0, "std": 3.0}  
DEFAULT_INF_AGENT_DIST = {"mean": 15.0, "std": 3.0}

# Optimization Hyperparameters for Coordinate Newton
NUM_OPTIMIZATION_EPOCHS = 20 # Number of full passes over all parameters
OPTIMIZATION_LEARNING_RATE = 0.5 # Learning rate for Newton steps
OPTIMIZATION_EPSILON = 1e-6    # Small constant for numerical stability in Newton step denominator

class NumpyFloatEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy float types."""
    def default(self, obj):
        if isinstance(obj, (np.floating, jnp.floating)):
            return float(obj)
        if isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            # Handle NaN for JSON compatibility
            if jnp.isscalar(obj) and jnp.isnan(obj):
                return None # or 'NaN' as a string, depending on preference
            return [None if jnp.isnan(x) else x for x in obj.tolist()] if obj.ndim > 0 else (None if jnp.isnan(obj.item()) else obj.item())
        if jnp.isnan(obj): # Handle scalar JAX NaN
            return None
        return json.JSONEncoder.default(self, obj)

def preprocess_attempts_to_matrix(attempts_list):
    """
    Convert a list of attempt dictionaries to an outcomes matrix.
    The matrix includes only problems that were solved by at least one agent,
    and only agents that solved at least one problem.
    The matrix coordinates correspond to indices in the returned problem and agent lists.
    Outcomes are 1 (solved), -1 (failed), or 0 (no_show).
    """
    # Identify the problems and agents that solved at least one problem
    problems = list({
        attempt["problem"] for attempt in attempts_list if attempt["outcome"] == "solved"
    })
    agents = list({
        attempt["agent"] for attempt in attempts_list if attempt["outcome"] == "solved"
    })
    problem_to_idx = {name: i for i, name in enumerate(problems)}
    agent_to_idx = {name: i for i, name in enumerate(agents)}
    n_problems = len(problems)
    n_agents = len(agents)

    # Populate the outcome matrix
    outcomes = np.full((n_problems, n_agents), 0, dtype=np.int8)
    for attempt in attempts_list: 
        problem = attempt["problem"]
        agent = attempt["agent"]
        if problem not in problem_to_idx or agent not in agent_to_idx:
            continue
        row_idx = problem_to_idx[problem]
        col_idx = agent_to_idx[agent]
        if attempt["outcome"] == "solved":
            outcomes[row_idx, col_idx] = 1
        elif attempt["outcome"] == "failed":
            outcomes[row_idx, col_idx] = -1

    return outcomes, problems, agents


def log_likelihood(outcomes, problem_difficulties_1d, agent_strengths_1d):
    """Calculate log-likelihood of outcomes given problem difficulties and agent strengths."""
    problem_difficulties_matrix = problem_difficulties_1d[:, None]
    agent_strengths_matrix = agent_strengths_1d[None, :]
    exponent_term = outcomes * (problem_difficulties_matrix - agent_strengths_matrix)
    log_probs_for_all_cells = -jnp.log1p(jnp.exp(exponent_term))
    actual_attempts_mask = (outcomes != 0)
    masked_log_probs = jnp.where(actual_attempts_mask, log_probs_for_all_cells, 0.0)
    return jnp.sum(masked_log_probs)

def negative_log_likelihood(params, outcomes, n_probs, n_agents, inf_prob_dist, inf_agent_dist):
    """Negative log-likelihood. This is the quantity we want to minimize."""
    problem_difficulties = lax.dynamic_slice(params, (0,), (n_probs,))
    agent_strengths = lax.dynamic_slice(params, (n_probs,), (n_agents,))
    prior_difficulties = -0.5 * jnp.sum((problem_difficulties - inf_prob_dist["mean"])**2 / inf_prob_dist["std"]**2)
    prior_strengths = -0.5 * jnp.sum((agent_strengths - inf_agent_dist["mean"])**2 / inf_agent_dist["std"]**2)
    ll = log_likelihood(outcomes, problem_difficulties, agent_strengths)
    return -(ll + prior_difficulties + prior_strengths)

def infer_parameters(outcomes, n_probs_for_inference, n_agents, inf_prob_dist, inf_agent_dist):
    """Maximum likelihood estimation using coordinate-wise Newton."""
    initialization_noise_std = 0.1 # Small noise to perturb initial values slightly

    # Initialize parameters close to the mean of their priors, with a little noise
    initial_problem_difficulties = np.random.normal(
        loc=inf_prob_dist["mean"], 
        scale=initialization_noise_std, 
        size=(n_probs_for_inference,)
    )
    initial_agent_strengths = np.random.normal(
        loc=inf_agent_dist["mean"], 
        scale=initialization_noise_std, 
        size=(n_agents,)
    )
    
    initial_params = jnp.concatenate([
        jnp.asarray(initial_problem_difficulties), 
        jnp.asarray(initial_agent_strengths)
    ])
    
    # Objective function to be passed to the optimizer
    # It needs to capture n_probs_for_inference, n_agents, inf_prob_dist, inf_agent_dist
    # because coordinate_newton_optimize only takes (params) -> scalar
    # However, our objective_fn in coordinate_newton_optimize is defined to match what this `objective` returns.
    # The `objective_fn` in the optimizer will be this one.
    def current_objective_fn_for_optimizer(params_vector):
        return negative_log_likelihood(params_vector, outcomes, n_probs_for_inference, n_agents, inf_prob_dist, inf_agent_dist)

    print("Starting custom coordinate-wise Newton optimization...")
    final_params, std_errors = coordinate_newton_optimize(
        current_objective_fn_for_optimizer, 
        initial_params, 
        n_probs_for_inference, # n_total_probs arg for optimizer (used for slicing by optimizer if needed)
        n_agents,              # n_total_agents arg for optimizer (used for slicing by optimizer if needed)
        NUM_OPTIMIZATION_EPOCHS, 
        OPTIMIZATION_LEARNING_RATE, 
        OPTIMIZATION_EPSILON
    )
    print("Custom optimization finished.")
    
    prob_std_errs = std_errors[:n_probs_for_inference]
    agent_std_errs = std_errors[n_probs_for_inference:]
    
    inferred_problem_difficulties = lax.dynamic_slice(final_params, (0,), (n_probs_for_inference,))
    inferred_agent_strengths = lax.dynamic_slice(final_params, (n_probs_for_inference,), (n_agents,))
    
    return inferred_problem_difficulties, inferred_agent_strengths, prob_std_errs, agent_std_errs

def main_rate():
    parser = argparse.ArgumentParser(description="Rate agents and problems from simulation attempts JSON.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file with attempts.")
    parser.add_argument("--output_file", type=str, default="ratings.json", help="Path to the output JSON file for ratings.")
    args = parser.parse_args()

    print(f"Loading attempts from: {args.input_file}")
    with open(args.input_file, 'r') as f:
        attempts_list_from_file = json.load(f)
    
    if not attempts_list_from_file:
        print("Error: Input file is empty or contains no attempts.")
        return

    agent_solve_counts = defaultdict(int)
    agent_attempt_counts = defaultdict(int)
    for attempt in attempts_list_from_file:
        agent_id_str = attempt["agent"]
        agent_attempt_counts[agent_id_str] += 1
        if attempt["outcome"] == "solved":
            agent_solve_counts[agent_id_str] += 1

    outcomes, problems, agents = preprocess_attempts_to_matrix(attempts_list_from_file)
    print(f"Data preprocessed: {len(problems)} problems, {len(agents)} agents found in input.")

    print("Starting Newton optimization...")
    inferred_problem_diffs, inferred_agent_strengths, prob_std_errors, agent_std_errors = infer_parameters(
        outcomes, 
        len(problems), 
        len(agents), 
        DEFAULT_INF_PROB_DIST, 
        DEFAULT_INF_AGENT_DIST
    )
    print("Inference complete.")

    # Create a list of agent data for sorting and printing
    agent_data = []
    for agent_idx in range(len(agents)):
        agent = agents[agent_idx]
        num_solved = agent_solve_counts.get(agent, 0)
        num_attempts = agent_attempt_counts.get(agent, 0)
        solved_per_attempt = (num_solved / num_attempts) if num_attempts > 0 else 0.0
        
        agent_data.append({
            "id": agent,
            "strength": inferred_agent_strengths[agent_idx],
            "std_error": agent_std_errors[agent_idx],
            "num_solved": num_solved,
            "num_attempts": num_attempts,
            "solved_per_attempt": solved_per_attempt
        })
    
    # Filter agents to include only those with num_solved > 0
    agents_with_solves = [agent for agent in agent_data if agent["num_solved"] > 0]

    # Sort filtered agents by inferred strength (descending) for printing and JSON ranking
    agents_with_solves.sort(key=lambda x: x["strength"], reverse=True)

    print("\n--- Agent Rankings (Agents with >0 solves) ---")
    header = f"{'Rank':>4} | {'ID':>4} | {'Strength':>12} | {'Solved':>6} | {'Attempts':>8} | {'Solve Rate':>10}"
    print(header)
    print("-" * len(header))

    agent_ratings_for_json = []
    for rank, agent_data in enumerate(agents_with_solves):
        se_str = f"{agent_data['std_error']:.2f}" if not (isinstance(agent_data['std_error'], float) and np.isnan(agent_data['std_error'])) else "N/A "
        strength_str = f"{agent_data['strength']:.2f} ± {se_str}"
        print(f"{rank+1:>4} | {agent_data['id']:>4} | {strength_str:>12} | {agent_data['num_solved']:>6} | {agent_data['num_attempts']:>8} | {agent_data['solved_per_attempt']:>10.3f}")
        
        agent_ratings_for_json.append({
            "rank": rank + 1,
            "id": agent_data['id'],
            "strength": agent_data['strength'],
            "std_error": agent_data['std_error'],
            "num_solved": agent_data['num_solved'],
            "num_attempts": agent_data['num_attempts'],
            "solved_per_attempt": agent_data['solved_per_attempt']
        })

    problem_ratings_for_json = []
    # Sort problems by inferred difficulty (descending)
    problem_print_data = []
    for problem_idx in range(len(problems)):
        problem = problems[problem_idx]
        
        # Find agents who solved this specific problem using the original outcomes_matrix
        solving_agent_indices = np.where(outcomes[problem_idx, :] == 1)[0]
        solving_agents_str = ", ".join(agents[idx] for idx in solving_agent_indices)
        problem_print_data.append({
            "id": problem,
            "difficulty": inferred_problem_diffs[problem_idx],
            "std_error": prob_std_errors[problem_idx],
            "solved_by": solving_agents_str
        })

    problem_print_data.sort(key=lambda x: x["difficulty"], reverse=True)
    
    # Minimal console print for problems, as it can be very long
    print("\n--- Top/Bottom Inferred Problem Difficulties --- ")
    header_prob = f"{'Rank':>4} | {'ID':>6} | {'Difficulty':>14} | {'Solved by':<20}"
    print(header_prob)
    print("-" * len(header_prob))
    for i, prob_data in enumerate(problem_print_data[:5]): # Top 5
        se_str = f"{prob_data['std_error']:.2f}" if not (isinstance(prob_data['std_error'], float) and np.isnan(prob_data['std_error'])) else "N/A "
        diff_str = f"{prob_data['difficulty']:.2f} ± {se_str}"
        print(f"{i+1:>4} | {prob_data['id']:>6} | {diff_str:>14} | {prob_data['solved_by']:<20}")
    if len(problem_print_data) > 10:
        print("...")
    for i, prob_data in enumerate(problem_print_data[-5:]): # Bottom 5 (least difficult of solved)
        rank_display = len(problem_print_data) - 5 + i +1
        se_str = f"{prob_data['std_error']:.2f}" if not (isinstance(prob_data['std_error'], float) and np.isnan(prob_data['std_error'])) else "N/A "
        diff_str = f"{prob_data['difficulty']:.2f} ± {se_str}"
        print(f"{rank_display:>4} | {prob_data['id']:>6} | {diff_str:>14} | {prob_data['solved_by']:<20}")

    # Prepare problem ratings for JSON
    for rank, prob_data in enumerate(problem_print_data):
        problem_ratings_for_json.append({
            "rank": rank + 1,
            "id": prob_data['id'],
            "difficulty": prob_data['difficulty'],
            "std_error": prob_data['std_error']
        })

    output_data = {
        "agents": agent_ratings_for_json, 
        "problems": problem_ratings_for_json,
        "inference_parameters_used": {
            "INF_PROB_DIST": DEFAULT_INF_PROB_DIST,
            "INF_AGENT_DIST": DEFAULT_INF_AGENT_DIST,
            "optimizer_settings": {
                "method": "coordinate_newton",
                "epochs": NUM_OPTIMIZATION_EPOCHS,
                "learning_rate": OPTIMIZATION_LEARNING_RATE,
                "epsilon": OPTIMIZATION_EPSILON
            }
        }
    }

    print(f"\nWriting ratings to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2, cls=NumpyFloatEncoder)
    print("Ratings successfully written.")

if __name__ == "__main__":
    main_rate() 