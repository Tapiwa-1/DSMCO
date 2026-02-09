from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import io

from models.matrix_engine import (
    calculate_transition_matrix,
    predict_next_state,
    calculate_expected_return,
    optimize_allocation,
)

app = Flask(__name__)
app.secret_key = 'your_secret_key' # Required for flash messages
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dsmco.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class OptimizationResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    formal_matrix_json = db.Column(db.Text)
    informal_matrix_json = db.Column(db.Text)
    formal_returns_json = db.Column(db.Text)
    informal_returns_json = db.Column(db.Text)
    risk_threshold = db.Column(db.Float)
    allocation_formal = db.Column(db.Float)
    portfolio_roa = db.Column(db.Float)
    portfolio_risk = db.Column(db.Float)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return upload()
    return render_template('index.html')

@app.route('/history')
def history():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 5, type=int)

    # Validate per_page
    if per_page not in [5, 10, 15, 20]:
        per_page = 5

    pagination = OptimizationResult.query.order_by(OptimizationResult.created_at.desc()).paginate(page=page, per_page=per_page)

    return render_template('history.html', pagination=pagination, per_page=per_page)

@app.route('/result/<int:id>')
def result(id):
    result = OptimizationResult.query.get_or_404(id)

    # Deserialize
    formal_matrix = pd.read_json(io.StringIO(result.formal_matrix_json))
    informal_matrix = pd.read_json(io.StringIO(result.informal_matrix_json))

    # Re-index if necessary (read_json might return string indices that sort alphabetically?)
    # matrix_engine uses: STATES = ['Performing', 'Delinquent', 'Defaulted', 'Recovered']
    # For now, we trust pd.read_json(to_json()) roundtrip preserves structure reasonably well for display.
    # But for guaranteed order, we might need to reindex.
    # Let's import STATES if we want to be strict, but for display likely fine.
    # Actually, let's just make sure the order is correct for the user.
    STATES = ['Performing', 'Delinquent', 'Defaulted', 'Recovered']
    formal_matrix = formal_matrix.reindex(index=STATES, columns=STATES)
    informal_matrix = informal_matrix.reindex(index=STATES, columns=STATES)

    return render_template(
        'results.html',
        formal_matrix_html=formal_matrix.to_html(classes='table table-striped'),
        informal_matrix_html=informal_matrix.to_html(classes='table table-striped'),
        allocation_formal=result.allocation_formal,
        portfolio_roa=result.portfolio_roa,
        portfolio_risk=result.portfolio_risk,
        result_id=result.id
    )

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
                formal_returns_np = np.array(formal_returns)
                informal_returns_np = np.array(informal_returns)

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
                formal_returns_np,
                informal_returns_np,
                risk_threshold
            )

            # Calculate Portfolio Metrics for display
            # 1. Predict Next State
            next_formal = predict_next_state(initial_dist, formal_matrix.values)
            next_informal = predict_next_state(initial_dist, informal_matrix.values)

            # 2. Returns
            ret_formal = calculate_expected_return(next_formal, formal_returns_np)
            ret_informal = calculate_expected_return(next_informal, informal_returns_np)

            portfolio_roa = allocation_formal * ret_formal + (1 - allocation_formal) * ret_informal

            # 3. Risk (Default Probability)
            prob_default_formal = next_formal[2] # Default is index 2
            prob_default_informal = next_informal[2]

            portfolio_risk = allocation_formal * prob_default_formal + (1 - allocation_formal) * prob_default_informal

            # Save to Database
            optimization_result = OptimizationResult(
                formal_matrix_json=formal_matrix.to_json(),
                informal_matrix_json=informal_matrix.to_json(),
                formal_returns_json=json.dumps(formal_returns_np.tolist()),
                informal_returns_json=json.dumps(informal_returns_np.tolist()),
                risk_threshold=risk_threshold,
                allocation_formal=float(allocation_formal),
                portfolio_roa=float(portfolio_roa),
                portfolio_risk=float(portfolio_risk)
            )
            db.session.add(optimization_result)
            db.session.commit()

            return render_template(
                'results.html',
                formal_matrix_html=formal_matrix.to_html(classes='table table-striped'),
                informal_matrix_html=informal_matrix.to_html(classes='table table-striped'),
                allocation_formal=allocation_formal,
                portfolio_roa=portfolio_roa,
                portfolio_risk=portfolio_risk,
                result_id=optimization_result.id
            )

        except Exception as e:
            flash(f"Error processing request: {e}")
            return redirect(url_for('index'))
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
