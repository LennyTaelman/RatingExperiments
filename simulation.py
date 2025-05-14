import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.optimize import minimize
from jax import hessian
import numpy as np

jax.config.update("jax_enable_x64", True)

# Strength and difficulty are measured in the 'natural' scale.
# In this scale, P(solve) = 1 / (1 + exp(difficulty - strength)).

# Reference solve probabilities for delta = difficulty - strength:
# delta =  2.197 => P(solve) = 0.100 
# delta =  4.394 => P(solve) = 0.010 
# delta =  6.908 => P(solve) = 0.001 

# to translate difficulty/strength to the chess ELO scale
# multiply by 138.6 (and add an arbitrary constant)

# Parameters for generating random problems and agents in the simulation phase
N_AGENTS = 20
NUM_PROBLEMS = 10000
MAX_DOWNTIME = 0.8
GEN_PROB_DIST = {"mean": 25.0, "std": 5.0}
GEN_AGENT_DIST = {"mean": 10.0, "std": 2.5}

# Parameters for maximum likelihood estimation in the inference phase (priors)
INF_PROB_DIST = {"mean": 30.0, "std": 10.0}  
INF_AGENT_DIST = {"mean": 10.0, "std": 1.0}

def generate_problems_and_agents(n_probs):
    """Generate problem difficulties and agent strengths."""
    problem_difficulties = np.random.normal(GEN_PROB_DIST["mean"], GEN_PROB_DIST["std"], size=(n_probs,))
    agent_strengths = np.random.normal(GEN_AGENT_DIST["mean"], GEN_AGENT_DIST["std"], size=(N_AGENTS,))
    agent_downtimes = np.random.uniform(0, MAX_DOWNTIME, size=(N_AGENTS,))
    return problem_difficulties, agent_strengths, agent_downtimes

def solve_probability(difficulty, strength):
    """Calculate probability of solving."""
    return 1 / (1 + jnp.exp(difficulty - strength))

def simulate_competition(problem_difficulties, agent_strengths, agent_downtimes):
    """Simulate competition."""
    difficulties_mat = problem_difficulties[:, None]
    strengths_mat = agent_strengths[None, :]
    solve_probs = solve_probability(difficulties_mat, strengths_mat)
    
    outcomes = np.random.uniform(size=(problem_difficulties.shape[0], N_AGENTS)) < solve_probs
    downtime_mask = np.random.uniform(size=(problem_difficulties.shape[0], N_AGENTS)) < agent_downtimes[None, :]
    outcomes = jnp.where(downtime_mask, 0, jnp.where(outcomes, 1, -1))
    return outcomes

def log_likelihood(outcomes, problem_difficulties_1d, agent_strengths_1d):
    """Calculate log-likelihood of outcomes given problem difficulties and agent strengths."""
    problem_difficulties_matrix = problem_difficulties_1d[:, None]
    agent_strengths_matrix = agent_strengths_1d[None, :]

    exponent_term = outcomes * (problem_difficulties_matrix - agent_strengths_matrix)
    log_probs_for_all_cells = -jnp.log1p(jnp.exp(exponent_term))
    
    actual_attempts_mask = (outcomes != 0)
    masked_log_probs = jnp.where(actual_attempts_mask, log_probs_for_all_cells, 0.0)
    return jnp.sum(masked_log_probs)

def negative_log_likelihood(params, outcomes, n_probs, n_agents):
    """Negative log-likelihood. This is the quantity we want to minimize."""
    problem_difficulties = lax.dynamic_slice(params, (0,), (n_probs,))
    agent_strengths = lax.dynamic_slice(params, (n_probs,), (n_agents,))
    
    prior_difficulties = -0.5 * jnp.sum((problem_difficulties - INF_PROB_DIST["mean"])**2 / INF_PROB_DIST["std"]**2)
    prior_strengths = -0.5 * jnp.sum((agent_strengths - INF_AGENT_DIST["mean"])**2 / INF_AGENT_DIST["std"]**2)
        
    ll = log_likelihood(outcomes, problem_difficulties, agent_strengths)
    return -(ll + prior_difficulties + prior_strengths)

def infer_parameters(outcomes, n_probs, n_agents):
    """Maximum likelihood estimation."""
    initial_problem_difficulties = np.random.normal(INF_PROB_DIST["mean"], INF_PROB_DIST["std"], size=(n_probs,))
    initial_agent_strengths = np.random.normal(INF_AGENT_DIST["mean"], INF_AGENT_DIST["std"], size=(n_agents,))
    
    initial_params = jnp.concatenate([initial_problem_difficulties, initial_agent_strengths])
    
    def objective(params):
        return negative_log_likelihood(params, outcomes, n_probs, n_agents)
    
    result = minimize(objective, initial_params, method='BFGS')
    H = hessian(objective)(result.x)
    
    try:
        cov = jnp.linalg.inv(H)
        std_errors = jnp.sqrt(jnp.diag(cov))
        problem_difficulty_std_errors = std_errors[:n_probs]
        agent_strength_std_errors = std_errors[-n_agents:]
    except:
        print("Warning: Could not compute standard errors (Hessian inversion failed)")
        problem_difficulty_std_errors = jnp.zeros(n_probs)
        agent_strength_std_errors = jnp.zeros(n_agents)
    
    inferred_problem_difficulties = lax.dynamic_slice(result.x, (0,), (n_probs,))
    inferred_agent_strengths = lax.dynamic_slice(result.x, (n_probs,), (n_agents,))
    
    return inferred_problem_difficulties, inferred_agent_strengths, problem_difficulty_std_errors, agent_strength_std_errors

def prune_unsolved_problems(outcomes, problem_difficulties):
    """Inputs/outputs are natural scale strengths/difficulties."""
    solved_mask = jnp.any(outcomes == 1, axis=1)
    pruned_outcomes = outcomes[solved_mask]
    pruned_problem_difficulties = problem_difficulties[solved_mask]
    return pruned_outcomes, pruned_problem_difficulties, solved_mask

def run_experiment(n_probs):
    """Run experiment. Returns natural scale strengths/difficulties and std errors."""
    true_problem_difficulties, true_agent_strengths, agent_downtimes = generate_problems_and_agents(n_probs)
    
    print(f"Agent Strengths (generated, natural scale), sorted: {[f'{s:.3f}' for s in jnp.sort(true_agent_strengths)]}")
    print(f"Agent downtimes: {[f'{dt:.2f}' for dt in agent_downtimes]}")
    print(f"Problem Difficulties (generated, natural scale, sample), sorted: {[f'{d:.3f}' for d in jnp.sort(true_problem_difficulties[:10])]}")
    
    outcomes = simulate_competition(true_problem_difficulties, true_agent_strengths, agent_downtimes)
    pruned_outcomes, pruned_true_problem_difficulties, solved_mask = prune_unsolved_problems(outcomes, true_problem_difficulties)
    n_solved_probs = jnp.sum(solved_mask)

    if n_solved_probs == 0:
        print("No problems solved, skipping...")
        return None
    
    min_diff_solved = jnp.min(pruned_true_problem_difficulties)
    max_diff_solved = jnp.max(pruned_true_problem_difficulties)
    median_diff_solved = jnp.median(pruned_true_problem_difficulties)
    
    inferred_problem_difficulties, inferred_agent_strengths, problem_std_errors, agent_std_errors = infer_parameters(pruned_outcomes, n_solved_probs, N_AGENTS)
    
    problems_solved_by_agent = jnp.sum(outcomes == 1, axis=0)
    
    return {
        'n_probs': n_probs,
        'n_solved_probs': n_solved_probs,
        'true_agent_strengths': true_agent_strengths,
        'inferred_agent_strengths': inferred_agent_strengths,
        'agent_strength_std_errors': agent_std_errors,
        'problems_solved': problems_solved_by_agent,
        'agent_downtimes': agent_downtimes,
        'solved_problem_stats': {
            'min_difficulty': min_diff_solved,
            'max_difficulty': max_diff_solved,
            'median_difficulty': median_diff_solved
        },
        'true_problem_difficulties_all': true_problem_difficulties,
        'inferred_problem_difficulties_solved': inferred_problem_difficulties,
        'problem_std_errors_solved': problem_std_errors,
        'solved_mask': solved_mask,
        'outcomes': outcomes
    }

def main():
    
    result = run_experiment(NUM_PROBLEMS)
    if result is None: return
        
    print("\n--- Agent Rankings ---\n")
    print(f"{'id':>3} | {'True Str':>8} | {'Inferred Str ':>12} | {'Solved':>6} | {'Rate':>6}")
    print("-" * 50)
    
    # Original results from inference
    inferred_agent_strengths_orig = result['inferred_agent_strengths']
    true_agent_strengths_res = result['true_agent_strengths'] 
    agent_strength_std_errors_res = result['agent_strength_std_errors']

    # Calculate adjustment to make mean inferred agent strength match generated mean
    mean_inferred_agent_strength = jnp.mean(inferred_agent_strengths_orig)
    target_mean_agent_strength = GEN_AGENT_DIST["mean"]
    adjustment_offset = target_mean_agent_strength - mean_inferred_agent_strength
    print(f"(Adjusting inferred strengths & difficulties by {adjustment_offset:.3f} to match mean generated agent strength of {target_mean_agent_strength:.3f})")

    # Apply adjustment to inferred values for display
    adjusted_inferred_agent_strengths = inferred_agent_strengths_orig + adjustment_offset
            
    sorted_agent_indices = jnp.argsort(-adjusted_inferred_agent_strengths) # Sort by adjusted inferred strength
    
    for agent_idx in sorted_agent_indices:
        count = result['problems_solved'][agent_idx]
        if count == 0 and MAX_DOWNTIME < 1.0 : continue
        
        true_strength_val = true_agent_strengths_res[agent_idx] 
        inferred_strength_val = adjusted_inferred_agent_strengths[agent_idx] # Use adjusted inferred strength
        std_err_val = agent_strength_std_errors_res[agent_idx] # Std error is for original inferred value
        downtime = result['agent_downtimes'][agent_idx]
        attempts = NUM_PROBLEMS * (1 - downtime)
        solve_rate = (count / attempts) * 1000 if attempts > 0 else 0
        print(f"{f'A{int(agent_idx)+1}':>3} | {true_strength_val:8.3f} | {inferred_strength_val:7.3f} ± {std_err_val:5.3f} | {int(count):6d} | {solve_rate:6.1f}")
    
    print("\n--- Problem Difficulties (random sample) ---\n")
    print(f"{'id':>6} | {'True Diff':>9} | {'Inferred Diff':>13} | Solved by")
    print("-" * 60)
    
    solved_mask_res = result['solved_mask']
    true_problem_difficulties_all_res = result['true_problem_difficulties_all']
    inferred_problem_difficulties_solved_orig = result['inferred_problem_difficulties_solved']
    problem_std_errors_solved_res = result['problem_std_errors_solved']

    # Apply the same adjustment to inferred problem difficulties
    adjusted_inferred_problem_difficulties_solved = inferred_problem_difficulties_solved_orig + adjustment_offset

    # True difficulties for solved problems (no adjustment)
    true_problem_difficulties_solved = true_problem_difficulties_all_res[solved_mask_res]
    
    n_solved = result['n_solved_probs']
    indices_of_solved_problems_in_inferred_array = jnp.arange(n_solved)

    n_to_show = min(20, n_solved)
    if n_to_show > 0:
        sampled_indices_within_solved = np.random.choice(indices_of_solved_problems_in_inferred_array, size=n_to_show, replace=False)
        # Sort based on adjusted inferred difficulties
        sampled_adjusted_inferred_difficulties = adjusted_inferred_problem_difficulties_solved[sampled_indices_within_solved]
        display_order_indices = sampled_indices_within_solved[jnp.argsort(-sampled_adjusted_inferred_difficulties)]
        
        original_problem_indices_all = jnp.where(solved_mask_res)[0]

        for solved_idx in display_order_indices: 
            original_idx = original_problem_indices_all[solved_idx] 
            
            true_diff_val = true_problem_difficulties_solved[solved_idx] 
            inferred_diff_val = adjusted_inferred_problem_difficulties_solved[solved_idx] # Use adjusted inferred difficulty
            std_err_val = problem_std_errors_solved_res[solved_idx] # Std error is for original inferred value
            
            solving_agents = jnp.where(result['outcomes'][original_idx] == 1)[0]
            solving_agents_str = ", ".join([f"A{j+1}" for j in solving_agents])
            
            print(f"{f'P{original_idx+1}':>6} | {true_diff_val:9.3f} | {inferred_diff_val:8.3f} ± {std_err_val:5.3f} | {solving_agents_str}")
    
    print(f"\nTotal solved problems: {n_solved} out of {NUM_PROBLEMS}")

if __name__ == "__main__":
    main() 