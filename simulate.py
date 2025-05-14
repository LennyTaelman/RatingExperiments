import numpy as np
import json
import argparse

# Parameters for generating problems and agents in the simulation phase
# These can be overridden by command-line arguments
DEFAULT_N_AGENTS = 20
DEFAULT_NUM_PROBLEMS = 1000 # Changed from 10000 for quicker testing, can be set by CLI
DEFAULT_MAX_DOWNTIME = 0.8
DEFAULT_GEN_PROB_DIST = {"mean": 25.0, "std": 5.0}
DEFAULT_GEN_AGENT_DIST = {"mean": 10.0, "std": 2.5}

def generate_problems_and_agents(n_probs, n_agents, gen_prob_dist, gen_agent_dist, max_downtime):
    """Generate problem difficulties and agent strengths."""
    problem_difficulties = np.random.normal(gen_prob_dist["mean"], gen_prob_dist["std"], size=(n_probs,))
    agent_strengths = np.random.normal(gen_agent_dist["mean"], gen_agent_dist["std"], size=(n_agents,))
    agent_downtimes = np.random.uniform(0, max_downtime, size=(n_agents,))
    return problem_difficulties, agent_strengths, agent_downtimes

def solve_probability(difficulty, strength):
    """Calculate probability of solving."""
    return 1 / (1 + np.exp(difficulty - strength)) # Using np.exp for numpy array compatibility

def simulate_competition_to_matrix(problem_difficulties, agent_strengths, agent_downtimes, n_agents):
    """Simulate competition and return an outcomes matrix (1=solve, -1=fail, 0=no_show)."""
    n_probs = problem_difficulties.shape[0]
    difficulties_mat = problem_difficulties[:, np.newaxis] # Shape: (n_probs, 1)
    strengths_mat = agent_strengths[np.newaxis, :]      # Shape: (1, n_agents)
    
    solve_probs = solve_probability(difficulties_mat, strengths_mat) # Shape: (n_probs, n_agents)
    
    # Generate outcomes based on solve_probs (True for solve, False for fail initially)
    solved_outcomes = np.random.uniform(size=(n_probs, n_agents)) < solve_probs
    
    # Generate downtime mask
    downtime_mask = np.random.uniform(size=(n_probs, n_agents)) < agent_downtimes[np.newaxis, :]
    
    # Combine into final outcomes: 1 for success, -1 for fail, 0 for no-show
    # Using np.where here, as jnp is not imported in this script
    outcomes = np.where(downtime_mask, 0, np.where(solved_outcomes, 1, -1))
    
    return outcomes.astype(np.int8) # Ensure integer types for outcomes

def format_attempts_from_matrix(outcomes_matrix):
    """Convert outcomes matrix to a list of attempt dictionaries, skipping no_shows."""
    attempts_list = []
    n_probs, n_agents = outcomes_matrix.shape
    
    for prob_idx in range(n_probs):
        for agent_idx in range(n_agents):
            outcome_val = outcomes_matrix[prob_idx, agent_idx]
            
            if outcome_val == 0:  # This is a no_show
                continue      # Skip recording this attempt
            
            if outcome_val == 1:
                outcome_str = "solved"
            elif outcome_val == -1:
                outcome_str = "failed"
            # No 'else' needed here, as we skipped outcome_val == 0
                
            attempts_list.append({
                "agent": f"A{agent_idx + 1}",
                "problem": f"P{prob_idx + 1}",
                "outcome": outcome_str
            })
    return attempts_list

def main_simulate():
    parser = argparse.ArgumentParser(description="Simulate a tournament and output attempts to JSON.")
    parser.add_argument("--output_file", type=str, default="simulation_attempts.json", help="Path to the output JSON file.")
    parser.add_argument("--num_problems", type=int, default=DEFAULT_NUM_PROBLEMS, help="Number of problems to simulate.")
    parser.add_argument("--num_agents", type=int, default=DEFAULT_N_AGENTS, help="Number of agents to simulate.")
    parser.add_argument("--max_downtime", type=float, default=DEFAULT_MAX_DOWNTIME, help="Maximum agent downtime probability.")
    # For dict arguments, it's a bit more complex, usually done via multiple args or json string
    # For simplicity, we'll use defaults here but acknowledge they could be CLI args
    args = parser.parse_args()

    print(f"Starting simulation with: {args.num_problems} problems, {args.num_agents} agents.")
    print(f"Generation - Problem Dist: Mean={DEFAULT_GEN_PROB_DIST['mean']}, Std={DEFAULT_GEN_PROB_DIST['std']}")
    print(f"Generation - Agent Dist: Mean={DEFAULT_GEN_AGENT_DIST['mean']}, Std={DEFAULT_GEN_AGENT_DIST['std']}")
    print(f"Max Downtime: {args.max_downtime}")


    problem_difficulties, agent_strengths, agent_downtimes = generate_problems_and_agents(
        args.num_problems, 
        args.num_agents, 
        DEFAULT_GEN_PROB_DIST, 
        DEFAULT_GEN_AGENT_DIST, 
        args.max_downtime
    )
    
    outcomes_matrix = simulate_competition_to_matrix(
        problem_difficulties, 
        agent_strengths, 
        agent_downtimes,
        args.num_agents 
    )
    
    attempts_list = format_attempts_from_matrix(outcomes_matrix)
    
    with open(args.output_file, 'w') as f:
        json.dump(attempts_list, f, indent=2)
        
    print(f"Simulation complete. {len(attempts_list)} attempts recorded.")
    print(f"Total problem-agent pairs: {args.num_problems * args.num_agents}")
    actual_attempts = sum(1 for attempt in attempts_list if attempt['outcome'] != 'no_show')
    solved_count = sum(1 for attempt in attempts_list if attempt['outcome'] == 'solved')
    print(f"Actual attempts (not no_show): {actual_attempts}")
    print(f"Solved attempts: {solved_count}")
    print(f"Output written to {args.output_file}")

if __name__ == "__main__":
    main_simulate() 