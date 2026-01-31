from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import random

@dataclass(frozen=True)
class SequenceModel:
    # kind is either "geometric" or "polynomial".
    kind: str

    # For polynomial: coefficients and degree are used.
    # y(n) = c0 + c1 * n + c2 * n^2 + ...
    coefficients: np.ndarray | None
    degree: int

    # For geometric: a and r are used.
    # y(n) = a * r^n
    a: float
    r: float

    # True if the model matches all input points within tolerance.
    exact_fit: bool


def build_design_matrix(indices: np.ndarray, degree: int) -> np.ndarray:
    """
    Build the Vandermonde-like matrix for polynomial regression.
    Each row is [1, n, n^2, ..., n^degree].
    """
    num_points = indices.shape[0]
    num_columns = degree + 1

    matrix = np.zeros((num_points, num_columns), dtype="float64")

    for column_index in range(num_columns):
        power = column_index
        matrix[:, column_index] = indices ** power

    return matrix


def try_fit_geometric(values: np.ndarray, tolerance: float) -> SequenceModel | None:
    """
    Try to fit a simple geometric model:
        y(n) = a * r^n, with n = 0, 1, 2, ...

    We require:
        - at least two points
        - no zero values (to avoid division problems)
        - consistent ratio y[n+1] / y[n] within tolerance
    
    If there are zeros, we return None and let polynomial handle it.
    """
    num_points = values.shape[0]

    if num_points < 2:
        return None

    index = 0
    while index < num_points:
        if abs(values[index]) < tolerance:
            return None
        index = index + 1

    first_ratio = values[1] / values[0]

    index = 1
    while index < num_points - 1:
        denominator = values[index]
        if abs(denominator) < tolerance:
            return None
        ratio = values[index + 1] / denominator
        if abs(ratio - first_ratio) > tolerance:
            return None
        index = index + 1

    a = float(values[0])
    r = float(first_ratio)

    index = 0
    while index < num_points:
        expected = a * (r ** index)
        error = abs(expected - values[index])
        if error > tolerance:
            return None
        index = index + 1

    return SequenceModel(
        kind="geometric",
        coefficients=None,
        degree=-1,
        a=a,
        r=r,
        exact_fit=True,
    )


def fit_lowest_degree_polynomial(
    values: np.ndarray,
    maximum_degree: int,
    tolerance: float,
) -> SequenceModel:
    """
    Try degrees from 0 up to maximum_degree and return the simplest
    polynomial that fits all points within the given tolerance.

    If none of the degrees fit exactly, we still return the degree
    with the smallest error, but mark exact_fit as False.
    """
    num_points = values.shape[0]

    indices = np.arange(num_points, dtype="float64")

    best_coefficients = None
    best_degree = 0
    best_error = None

    degree = 0
    while degree <= maximum_degree and degree < num_points:
        design = build_design_matrix(indices, degree)
        solution = np.linalg.lstsq(design, values, rcond=None)
        coefficients = solution[0]

        predictions = np.dot(design, coefficients)
        errors = predictions - values
        maximum_absolute_error = float(np.max(np.abs(errors)))

        if best_error is None or maximum_absolute_error < best_error:
            best_error = maximum_absolute_error
            best_degree = degree
            best_coefficients = coefficients

        if maximum_absolute_error <= tolerance:
            return SequenceModel(
                kind="polynomial",
                coefficients=coefficients,
                degree=degree,
                a=0.0,
                r=0.0,
                exact_fit=True,
            )

        degree = degree + 1

    return SequenceModel(
        kind="polynomial",
        coefficients=best_coefficients,
        degree=best_degree,
        a=0.0,
        r=0.0,
        exact_fit=False,
    )


def simplify_coefficient(value: float) -> float:
    """
    Clean up tiny numerical noise by rounding.
    """
    rounded = round(value, 10)
    if abs(rounded) < 1e-10:
        return 0.0
    return rounded


def format_polynomial(model: SequenceModel) -> str:
    """
    Turn polynomial coefficients into a human-readable formula in n.
    """
    if model.coefficients is None:
        return "0"

    parts = []

    index = 0
    while index < model.coefficients.shape[0]:
        raw_coefficient = model.coefficients[index]
        coefficient = simplify_coefficient(float(raw_coefficient))

        power = index

        if coefficient != 0.0:
            term = ""

            if power == 0:
                term = str(coefficient)
            else:
                if coefficient == 1.0:
                    term = "n"
                elif coefficient == -1.0:
                    term = "-n"
                else:
                    term = str(coefficient) + " * n"

                if power > 1:
                    term = term + "^" + str(power)

            parts.append(term)

        index = index + 1

    if len(parts) == 0:
        return "0"

    formula = " + ".join(parts)
    formula = formula.replace("+ -", "- ")

    return formula


def format_geometric(model: SequenceModel) -> str:
    """
    Turn geometric parameters into a human-readable formula in n.
    """
    a_clean = simplify_coefficient(model.a)
    r_clean = simplify_coefficient(model.r)

    if a_clean == 0.0:
        return "0"

    if r_clean == 1.0:
        return str(a_clean)

    return str(a_clean) + " * " + str(r_clean) + "^n"


def format_model(model: SequenceModel) -> str:
    """
    Dispatch to the appropriate formatter based on model kind.
    """
    if model.kind == "geometric":
        return format_geometric(model)
    return format_polynomial(model)


def predict_next_values(
    model: SequenceModel,
    number_of_existing_points: int,
    number_of_future_points: int,
) -> np.ndarray:
    """
    Use the chosen model to predict future values.
    """
    if number_of_future_points <= 0:
        return np.zeros((0,), dtype="float64")

    if model.kind == "geometric":
        start_index = number_of_existing_points
        end_index = number_of_existing_points + number_of_future_points

        indices = np.arange(start_index, end_index, dtype="float64")

        predictions = model.a * (model.r ** indices)
        return predictions

    start_index = number_of_existing_points
    end_index = number_of_existing_points + number_of_future_points

    indices = np.arange(start_index, end_index, dtype="float64")

    design = build_design_matrix(indices, model.degree)

    predictions = np.dot(design, model.coefficients)

    return predictions


def fit_best_model(values: np.ndarray, kind: str = "auto") -> SequenceModel:
    """
    Try models and pick the simplest one that works.

    If kind is:
        - "geometric": try geometric first, then fall back to polynomial.
        - "polynomial": only try polynomial.
        - "auto": try geometric, then polynomial.
    """
    tolerance = 1e-6

    if kind in ("geometric", "auto"):
        geometric_model = try_fit_geometric(values, tolerance=tolerance)
        if geometric_model is not None:
            return geometric_model

    maximum_degree = 6
    polynomial_model = fit_lowest_degree_polynomial(
        values=values,
        maximum_degree=maximum_degree,
        tolerance=tolerance,
    )
    return polynomial_model


def generate_random_model(kind: str = "auto", max_degree: int = 2) -> SequenceModel:
    """
    Generate a random sequence model (either geometric or polynomial).
    """
    if kind == "auto":
        kind = random.choice(["polynomial", "geometric"])

    if kind == "geometric":
        # y(n) = a * r^n
        # Keep it simple: integers or simple fractions
        a = float(random.randint(1, 5) * random.choice([1, -1]))
        r = random.choice([2.0, 3.0, 0.5, -2.0, -0.5])
        
        return SequenceModel(
            kind="geometric",
            coefficients=None,
            degree=-1,
            a=a,
            r=r,
            exact_fit=True,
        )
    else:
        # Polynomial
        degree = random.randint(0, max_degree)
        coeffs = np.zeros(degree + 1)
        for i in range(degree + 1):
            coeffs[i] = random.randint(-3, 3)
        
        # Ensure highest degree term is non-zero if degree > 0
        if degree > 0 and coeffs[-1] == 0:
            coeffs[-1] = random.choice([1, -1])
            
        return SequenceModel(
            kind="polynomial",
            coefficients=coeffs,
            degree=degree,
            a=0.0,
            r=0.0,
            exact_fit=True,
        )
