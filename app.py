from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np

from models.matrix_engine import (
    calculate_transition_matrix,
    predict_next_state,
    calculate_expected_return,
    optimize_allocation,
)

app = Flask(__name__)
app.secret_key = 'your_secret_key' # Required for flash messages

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return upload()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        try:
            # Read CSV
            df = pd.read_csv(file)

            # Validate columns
            required_cols = ['Loan_ID', 'Period', 'State', 'Sector']
            if not all(col in df.columns for col in required_cols):
                flash(f"CSV must contain columns: {', '.join(required_cols)}")
                return redirect(url_for('index'))

            # Calculate Matrices
            formal_matrix = calculate_transition_matrix(df, sector='Formal')
            informal_matrix = calculate_transition_matrix(df, sector='Informal')

            # Parse Returns
            try:
                formal_returns = [float(x.strip()) for x in request.form['formal_returns'].split(',')]
                informal_returns = [float(x.strip()) for x in request.form['informal_returns'].split(',')]

                if len(formal_returns) != 4 or len(informal_returns) != 4:
                    raise ValueError("Returns must have exactly 4 values.")

                # Convert to numpy arrays
                formal_returns = np.array(formal_returns)
                informal_returns = np.array(informal_returns)

                risk_threshold = float(request.form['risk_threshold'])

            except ValueError as e:
                flash(f"Invalid input for returns or threshold: {e}")
                return redirect(url_for('index'))

            # Initial Distribution (Assume 100% Performing for simplicity)
            initial_dist = np.array([1.0, 0.0, 0.0, 0.0])

            # Optimize
            allocation_formal = optimize_allocation(
                formal_matrix.values,
                informal_matrix.values,
                initial_dist,
                initial_dist,
                formal_returns,
                informal_returns,
                risk_threshold
            )

            # Calculate Portfolio Metrics for display
            # 1. Predict Next State
            next_formal = predict_next_state(initial_dist, formal_matrix.values)
            next_informal = predict_next_state(initial_dist, informal_matrix.values)

            # 2. Returns
            ret_formal = calculate_expected_return(next_formal, formal_returns)
            ret_informal = calculate_expected_return(next_informal, informal_returns)

            portfolio_roa = allocation_formal * ret_formal + (1 - allocation_formal) * ret_informal

            # 3. Risk (Default Probability)
            prob_default_formal = next_formal[2] # Default is index 2
            prob_default_informal = next_informal[2]

            portfolio_risk = allocation_formal * prob_default_formal + (1 - allocation_formal) * prob_default_informal

            return render_template(
                'results.html',
                formal_matrix_html=formal_matrix.to_html(classes='table table-striped'),
                informal_matrix_html=informal_matrix.to_html(classes='table table-striped'),
                allocation_formal=allocation_formal,
                portfolio_roa=portfolio_roa,
                portfolio_risk=portfolio_risk
            )

        except Exception as e:
            flash(f"Error processing request: {e}")
            return redirect(url_for('index'))
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
