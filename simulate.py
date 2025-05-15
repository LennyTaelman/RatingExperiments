import numpy as np
import json
import argparse

# Parameters for generating problems and agents in the simulation phase
NUM_AGENTS = 20
NUM_PROBLEMS = 1000 

# Probabilistic model for simulating solution attempts:
# Agent has a strength s, problem has a difficulty d,
# Success proability = 1 / (1 + exp(delta)), with delta = d - s

# delta = 0 => P(success) = 1/2 
# delta = 4.59 => P(success) = 1/100 
# delta = 9.21 => P(success) = 1/10000


# problems and agents are normally distributed with the following parameters
MEAN_DELTA = 15 # Mean value of delta = d - s 
PROBLEM_DIFFICULTY_STD = 5 # Standard deviation for problem difficulty d
AGENT_STRENGTH_STD = 2.5 # Standard deviation for agent strength s
MAX_DOWNTIME = 0.8 # Each agent is down for random fraction between 0 and MAX_DOWNTIME


def solve_probability(d, s):
    """Calculate probability of solving."""
    return 1 / (1 + np.exp(d - s)) # Using np.exp for numpy array compatibility


def generate_problems_and_agents(
        n_probs,
        n_agents,
        mean_delta,
        problem_difficulty_std,
        agent_strength_std,
        max_downtime):
    """Generate problem difficulties and agent strengths."""
    d = np.random.normal(mean_delta, problem_difficulty_std, size=(n_probs,))
    s = np.random.normal(0, agent_strength_std, size=(n_agents,))
    agent_downtimes = np.random.uniform(0, max_downtime, size=(n_agents,))
    return d, s, agent_downtimes


def simulate_competition(problem_difficulties, agent_strengths, agent_downtimes):
    """Simulate competition and return an outcomes matrix (1=solve, -1=fail, 0=no_show)."""
    n_probs = problem_difficulties.shape[0]
    n_agents = agent_strengths.shape[0]

    difficulties_mat = problem_difficulties[:, np.newaxis] # Shape: (n_probs, 1)
    strengths_mat = agent_strengths[np.newaxis, :]      # Shape: (1, n_agents)
    solve_probs = solve_probability(difficulties_mat, strengths_mat) # Shape: (n_probs, n_agents)
    
    # Generate outcomes based on solve_probs and agent downtime
    solved_outcomes = np.random.uniform(size=(n_probs, n_agents)) < solve_probs
    downtime_mask = np.random.uniform(size=(n_probs, n_agents)) < agent_downtimes[np.newaxis, :]
    outcomes = np.where(downtime_mask, 0, np.where(solved_outcomes, 1, -1))
    
    return outcomes.astype(np.int8) 

def list_attempts(outcomes):
    """Convert outcomes matrix to a list of attempt dictionaries, skipping no-shows."""
    attempts_list = []
    n_probs, n_agents = outcomes.shape
    
    for prob_idx in range(n_probs):
        for agent_idx in range(n_agents):
            outcome = outcomes[prob_idx, agent_idx]           
            if outcome == 0:  
                continue      # Skip no-shows
            if outcome == 1:
                outcome_str = "solved"
            elif outcome == -1:
                outcome_str = "failed"
            attempts_list.append({
                "agent": f"A{agent_idx + 1}",
                "problem": f"P{prob_idx + 1}",
                "outcome": outcome_str
            })
    return attempts_list

def main_simulate():
    parser = argparse.ArgumentParser(description="Simulate a tournament and output attempts to JSON.")
    parser.add_argument("--output_file", type=str, default="simulation_attempts.json", help="Path to the output JSON file.")
    args = parser.parse_args()

    problem_difficulties, agent_strengths, agent_downtimes = generate_problems_and_agents(
        NUM_PROBLEMS, 
        NUM_AGENTS, 
        MEAN_DELTA, 
        PROBLEM_DIFFICULTY_STD, 
        AGENT_STRENGTH_STD, 
        MAX_DOWNTIME
    )
    
    outcomes = simulate_competition(
        problem_difficulties, 
        agent_strengths, 
        agent_downtimes
    )
    
    attempts_list = list_attempts(outcomes)
    
    with open(args.output_file, 'w') as f:
        json.dump(attempts_list, f, indent=2)
        
    # Compute total number of unique problems that were solved by at least one agent
    num_problems_solved = np.sum(np.any(outcomes == 1, axis=1))
    print(f"Total number of unique problems solved: {num_problems_solved} out of {NUM_PROBLEMS}")


if __name__ == "__main__":
    main_simulate() 