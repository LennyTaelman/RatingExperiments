import jax
import jax.numpy as jnp

def coordinate_newton_optimize(objective_fn, initial_params, n_total_probs, n_total_agents, num_epochs, learning_rate, grad_hess_epsilon):
    """
    Performs coordinate-wise Newton optimization.
    Returns optimized parameters and their coordinate-wise standard errors.

    Args:
        objective_fn: A function that takes the full parameter vector and returns a scalar loss.
        initial_params: JAX array of initial parameters (concatenated problem difficulties and agent strengths).
        n_total_probs: The number of problem parameters in initial_params (used for slicing, not directly in objective_fn here).
        n_total_agents: The number of agent parameters in initial_params (used for slicing, not directly in objective_fn here).
        num_epochs: Number of full passes over all parameters.
        learning_rate: Damping factor for the Newton step.
        grad_hess_epsilon: Small value added to the second derivative for stability.

    Returns:
        Final optimized parameters (JAX array) and their coordinate-wise standard errors.
    """
    current_params = initial_params
    num_params = len(current_params)

    print(f"Starting coordinate-wise Newton optimization with {num_epochs} epochs.")
    print(f"  Learning rate: {learning_rate}, Epsilon for 2nd derivative: {grad_hess_epsilon}")

    for epoch in range(num_epochs):
        epoch_objective_start = objective_fn(current_params)
        for i in range(num_params):
            # Define a 1D objective function for the i-th parameter
            def f_i(p_i_scalar):
                # Create a new params vector where only the i-th element is p_i_scalar
                updated_params_for_f_i = current_params.at[i].set(p_i_scalar)
                return objective_fn(updated_params_for_f_i)
            
            # Current value of the parameter we are optimizing
            p_i_current_val = current_params[i]

            # Compute 1D derivatives
            grad_f_i = jax.grad(f_i)(p_i_current_val)
            # jax.grad(jax.grad(f_i)) is f_i'' for a scalar function f_i of a scalar input
            hess_f_i = jax.grad(jax.grad(f_i))(p_i_current_val) 

            # Newton Step for the i-th parameter
            if jnp.abs(hess_f_i) < grad_hess_epsilon: # Avoid division by zero or too small hessian
                # Fallback to gradient descent if second derivative is too small or wrong sign (for minimization)
                # Or simply skip update for this param in this step if hess_f_i is not reliably positive
                # For simplicity here, we use a larger epsilon check for the denominator directly
                delta = learning_rate * (grad_f_i / (jnp.sign(hess_f_i) * jnp.maximum(jnp.abs(hess_f_i), grad_hess_epsilon) + grad_hess_epsilon)) 
            else:
                 delta = learning_rate * (grad_f_i / (hess_f_i + jnp.sign(hess_f_i) * grad_hess_epsilon )) # Add epsilon to avoid zero, preserve sign
            
            current_params = current_params.at[i].add(-delta)
        
        epoch_objective_end = objective_fn(current_params)
        print(f"  Epoch {epoch + 1}/{num_epochs} completed. Objective: {epoch_objective_end:.2f} (Start: {epoch_objective_start:.2f}, Change: {epoch_objective_end - epoch_objective_start:.2f})")

    print("Coordinate-wise Newton optimization finished.")
    
    # Calculate coordinate-wise standard errors from the diagonal of the Hessian (second derivatives)
    std_errors_coordwise = jnp.zeros_like(current_params)
    print("Calculating coordinate-wise standard errors...")
    for i in range(num_params):
        def f_i_final(p_i_scalar):
            updated_params_for_f_i = current_params.at[i].set(p_i_scalar)
            return objective_fn(updated_params_for_f_i)
        
        hess_f_i_final = jax.grad(jax.grad(f_i_final))(current_params[i])
        
        if hess_f_i_final > grad_hess_epsilon: # Ensure second derivative is positive and reasonably large
            # Add epsilon before sqrt for robustness, especially if hess_f_i_final is small but positive
            std_errors_coordwise = std_errors_coordwise.at[i].set(1.0 / jnp.sqrt(hess_f_i_final + grad_hess_epsilon))
        else:
            # If 2nd derivative is not positive or too small, SE is undefined or very large.
            # Setting to a large number or NaN could be options. Here, we leave it as 0 (or np.inf / jnp.nan).
            std_errors_coordwise = std_errors_coordwise.at[i].set(jnp.nan) # Indicate undefined SE
            print(f"  Warning: Could not compute std error for param {i} (2nd deriv: {hess_f_i_final:.2e}). Setting to NaN.")

    print("Standard error calculation finished.")
    return current_params, std_errors_coordwise 