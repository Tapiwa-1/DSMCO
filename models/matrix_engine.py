import pandas as pd
import numpy as np

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
