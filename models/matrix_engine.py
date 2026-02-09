import pandas as pd
import numpy as np
from scipy.optimize import minimize

STATES = ['Performing', 'Delinquent', 'Defaulted', 'Recovered']

def calculate_transition_matrix(data: pd.DataFrame, sector: str = None) -> pd.DataFrame:
    """
    Calculates a 4x4 transition probability matrix from loan history data.

    Parameters:
    - data (pd.DataFrame): DataFrame containing loan history.
        Expected columns:
        - 'Loan_ID': Unique identifier for the loan.
        - 'Period': Time period (e.g., date or integer).
        - 'State': Current state of the loan.
        - 'Sector': (Optional) Sector of the loan ('Formal' or 'Informal').
    - sector (str, optional): If provided, filters data by this sector.

    Returns:
    - pd.DataFrame: 4x4 transition probability matrix.
    """
    df = data.copy()

    # Check required columns
    required_cols = ['Loan_ID', 'Period', 'State']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Filter by sector if provided
    if sector is not None:
        if 'Sector' not in df.columns:
            raise ValueError("Input DataFrame must contain 'Sector' column to filter by sector.")
        df = df[df['Sector'] == sector]

    if df.empty:
        # Return identity matrix if no data
        return pd.DataFrame(np.eye(4), index=STATES, columns=STATES)

    # Sort by Loan_ID and Period
    df = df.sort_values(by=['Loan_ID', 'Period'])

    # Create Next_State column
    # Shift State by -1 within each Loan_ID group
    # Note: Shift(-1) on the group ensures we don't cross Loan_IDs
    df['Next_State'] = df.groupby('Loan_ID')['State'].shift(-1)

    # Filter out rows where Next_State is NaN (last period for each loan)
    transitions = df.dropna(subset=['Next_State'])

    # Count transitions
    transition_counts = pd.crosstab(
        transitions['State'],
        transitions['Next_State']
    ).reindex(index=STATES, columns=STATES, fill_value=0)

    # Normalize rows
    row_sums = transition_counts.sum(axis=1)

    # Avoid division by zero
    transition_matrix = transition_counts.div(row_sums, axis=0)

    # Handle NaNs (where row sum was 0)
    for state in STATES:
        if row_sums[state] == 0:
            transition_matrix.loc[state, state] = 1.0 # Absorbing state
            transition_matrix.loc[state] = transition_matrix.loc[state].fillna(0.0)
        else:
             transition_matrix.loc[state] = transition_matrix.loc[state].fillna(0.0)

    return transition_matrix


def predict_next_state(current_state_distribution: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    """
    Predicts the next state distribution using Markov Chain formula: pi(t+1) = pi(t) * P

    Parameters:
    - current_state_distribution (np.ndarray): 1x4 array representing state probabilities at time t.
    - transition_matrix (np.ndarray): 4x4 transition probability matrix.

    Returns:
    - np.ndarray: 1x4 array representing predicted state probabilities at time t+1.
    """
    # Ensure inputs are numpy arrays
    pi_t = np.asarray(current_state_distribution)
    P = np.asarray(transition_matrix)

    # Calculate next state: pi(t+1) = pi(t) @ P
    pi_next = pi_t @ P

    return pi_next


def calculate_expected_return(state_distribution: np.ndarray, state_returns: np.ndarray) -> float:
    """
    Calculates the expected return for a sector based on its state distribution.
    R(t) = sum(pi_i(t) * r_i)

    Parameters:
    - state_distribution (np.ndarray): 1x4 array of state probabilities.
    - state_returns (np.ndarray): 1x4 array of returns for each state [Performing, Delinquent, Defaulted, Recovered].

    Returns:
    - float: Expected return.
    """
    return np.dot(state_distribution, state_returns)


def optimize_allocation(
    formal_matrix: np.ndarray,
    informal_matrix: np.ndarray,
    initial_formal_dist: np.ndarray,
    initial_informal_dist: np.ndarray,
    formal_returns: np.ndarray,
    informal_returns: np.ndarray,
    risk_threshold: float,
    risk_measure: str = 'default_prob'
) -> float:
    """
    Finds the optimal capital allocation to the Formal sector (x_F) to maximize total ROA,
    subject to a risk constraint.

    Maximize: ROA_t = x_F * R_F + (1 - x_F) * R_I
    Subject to: Risk(t) <= risk_threshold
                0 <= x_F <= 1

    Parameters:
    - formal_matrix, informal_matrix: 4x4 transition matrices.
    - initial_formal_dist, initial_informal_dist: 1x4 initial state distributions.
    - formal_returns, informal_returns: 1x4 arrays of returns per state.
    - risk_threshold: Maximum allowed risk.
    - risk_measure: Type of risk measure. Currently supports 'default_prob' (probability of Default state S3).

    Returns:
    - float: Optimal allocation to Formal sector (between 0 and 1).
    """

    # 1. Predict next state distributions for t+1
    pi_next_formal = predict_next_state(initial_formal_dist, formal_matrix)
    pi_next_informal = predict_next_state(initial_informal_dist, informal_matrix)

    # 2. Calculate expected returns for t+1
    # Note: Optimization is usually done based on expected future returns, so we use pi_next.
    # If the user meant current returns, they would pass current dist. Assuming t+1 optimization.
    R_formal = calculate_expected_return(pi_next_formal, formal_returns)
    R_informal = calculate_expected_return(pi_next_informal, informal_returns)

    # 3. Define Objective Function (Minimize Negative ROA)
    # x = allocation to Formal
    def objective(x):
        # maximize: x * R_F + (1-x) * R_I
        # minimize: -(x * R_F + (1-x) * R_I)
        return -1 * (x[0] * R_formal + (1 - x[0]) * R_informal)

    # 4. Define Risk Constraint
    # Risk <= Threshold  =>  Threshold - Risk >= 0
    def constraint_risk(x):
        alloc_formal = x[0]
        alloc_informal = 1 - alloc_formal

        if risk_measure == 'default_prob':
            # Risk is weighted probability of being in Default state (index 2 in STATES)
            # Default index is 2: ['Performing', 'Delinquent', 'Defaulted', 'Recovered']
            prob_default_formal = pi_next_formal[2]
            prob_default_informal = pi_next_informal[2]

            portfolio_risk = alloc_formal * prob_default_formal + alloc_informal * prob_default_informal
            return risk_threshold - portfolio_risk
        else:
            # Placeholder for other risk measures (e.g., standard deviation)
            # For now, default to 0 (always satisfied) or raise error
            return 0.0

    # 5. Constraints and Bounds
    constraints = [{'type': 'ineq', 'fun': constraint_risk}]
    bounds = [(0.0, 1.0)] # 0 <= x_F <= 1

    # Initial guess: 50% allocation
    x0 = [0.5]

    # 6. Run Optimization
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return float(result.x[0])
    else:
        # Fallback or error handling?
        # If optimization fails (e.g., no feasible solution), returning closest bound or raising error.
        # Often happens if risk constraint cannot be satisfied even with 0% or 100%.
        # Let's return the input guess or 0 if it failed due to strict constraints?
        # Better to return the best effort found, but result.x might be out of bounds if failed?
        # SLSQP usually respects bounds.
        # If no feasible solution exists (e.g. min risk > threshold), result.success is False.
        # We should probably raise an error or return NaN to indicate failure.
        # Let's return result.x[0] but log/warn. For now just return it.
        return float(result.x[0])
