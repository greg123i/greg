"""
Autoregressive-style sequence predictor and formula finder.
Entry point for the autoregressor tool.
"""

from __future__ import annotations

import sys
import numpy as np
from typing import Tuple, List, Optional

# Add project root to sys.path to ensure lib can be imported
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from lib.autoregressor import (
    SequenceModel,
    fit_best_model,
    format_model,
    predict_next_values,
)


def parse_list_of_numbers(text: str) -> List[float]:
    """
    Parse a single list of numbers from text.
    Handles spaces, commas, or both as separators.
    """
    numbers = []
    text = text.strip()
    if not text:
        return numbers

    parts = text.replace(",", " ").split()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            number = float(part)
            numbers.append(number)
        except ValueError:
            continue
    return numbers


def parse_sequence_of_lists(text: str) -> List[List[float]]:
    """
    Parse a sequence of lists from text.
    Lists can be separated by | or newlines.
    """
    lists = []
    
    if "|" in text:
        parts = text.split("|")
    else:
        parts = text.split("\n")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        numbers = parse_list_of_numbers(part)
        if numbers:
            lists.append(numbers)
            
    return lists


def parse_arguments(arguments: List[str]) -> Tuple[int, str, List[List[float]]]:
    """
    Parse command-line arguments.
    """
    future = 5
    kind = "auto"
    sequence_from_file = None
    
    # Simple argument parsing
    i = 0
    while i < len(arguments):
        arg = arguments[i]
        
        if arg == "--future":
            if i + 1 < len(arguments):
                future = int(arguments[i + 1])
                i += 2
                continue
        elif arg.startswith("--future="):
            future = int(arg.split("=")[1])
            i += 1
            continue
            
        if arg == "--kind":
            if i + 1 < len(arguments):
                kind = arguments[i + 1]
                i += 2
                continue
        elif arg.startswith("--kind="):
            kind = arg.split("=")[1]
            i += 1
            continue
            
        if arg == "--file":
            if i + 1 < len(arguments):
                path = arguments[i + 1]
                with open(path, "r", encoding="utf-8") as f:
                    sequence_from_file = parse_sequence_of_lists(f.read())
                i += 2
                continue
        elif arg.startswith("--file="):
            path = arg.split("=")[1]
            with open(path, "r", encoding="utf-8") as f:
                sequence_from_file = parse_sequence_of_lists(f.read())
            i += 1
            continue
            
        break
        
    remaining = arguments[i:]
    
    if sequence_from_file:
        return future, kind, sequence_from_file
        
    if not remaining:
        print("Usage: python -m ... [options] <numbers>")
        sys.exit(1)
        
    # Parse remaining args as sequence
    combined = " ".join(remaining)
    if "|" in combined:
        return future, kind, parse_sequence_of_lists(combined)
    
    # Single list case
    nums = parse_list_of_numbers(combined)
    return future, kind, [[x] for x in nums]


def run_sequence_demo(lists: List[List[float]], future_count: int, kind: str) -> None:
    """
    Fit models for each position in the lists and print results.
    """
    if not lists:
        print("No data provided.")
        return

    list_length = len(lists[0])
    # Validate lengths
    for lst in lists:
        if len(lst) != list_length:
            print("Error: All lists must have same length.")
            return

    print(f"Input: {len(lists)} lists of length {list_length}")
    
    for pos in range(list_length):
        # Extract column
        values = np.array([lst[pos] for lst in lists], dtype=float)
        
        model = fit_best_model(values, kind=kind)
        formula = format_model(model)
        
        print(f"\nPosition {pos}:")
        print(f"  Formula: y(n) = {formula}")
        
        future = predict_next_values(model, len(lists), future_count)
        future_str = " ".join([f"{v:.4f}" for v in future])
        print(f"  Prediction: {future_str}")


def draw_ui(screen, font, big_font, state):
    """Draw the UI elements."""
    screen.fill((30, 30, 30))
    y = 20
    
    # Input
    screen.blit(font.render("Enter sequence (space separated):", True, (200, 200, 200)), (20, y))
    y += 30
    
    pygame = sys.modules['pygame']
    pygame.draw.rect(screen, (50, 50, 50), (20, y, 920, 40))
    screen.blit(big_font.render(state['text'], True, (255, 255, 255)), (30, y + 10))
    y += 60
    
    # Results
    if state['error']:
        screen.blit(font.render(state['error'], True, (255, 100, 100)), (20, y))
    else:
        if state['formula']:
            screen.blit(big_font.render(state['formula'], True, (100, 255, 100)), (20, y))
            y += 40
            screen.blit(big_font.render(state['prediction'], True, (100, 200, 255)), (20, y))


def process_ui_input(text: str) -> dict:
    """Process input text and return result state."""
    state = {'text': text, 'error': '', 'formula': '', 'prediction': ''}
    
    try:
        # Assume single list input means a sequence
        nums = parse_list_of_numbers(text)
        if len(nums) < 2:
            state['error'] = "Need at least 2 numbers."
            return state
            
        # Treat as a sequence (column vector)
        values = np.array(nums, dtype=float)
        model = fit_best_model(values, kind="auto")
        
        state['formula'] = f"y(n) = {format_model(model)}"
        future = predict_next_values(model, len(nums), 5)
        state['prediction'] = "Next: " + " ".join([f"{v:.4f}" for v in future])
        
    except Exception as e:
        state['error'] = str(e)
        
    return state


def run_pygame_ui() -> None:
    """Run the Pygame UI."""
    try:
        import pygame
    except ImportError:
        print("Please install pygame.")
        return

    pygame.init()
    screen = pygame.display.set_mode((960, 540))
    pygame.display.set_caption("Sequence Finder")
    
    font = pygame.font.SysFont("consolas", 18)
    big_font = pygame.font.SysFont("consolas", 22)
    
    state = {'text': '', 'error': '', 'formula': '', 'prediction': ''}
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN:
                    # Process
                    res = process_ui_input(state['text'])
                    state.update(res)
                elif event.key == pygame.K_BACKSPACE:
                    state['text'] = state['text'][:-1]
                else:
                    state['text'] += event.unicode
                    
        draw_ui(screen, font, big_font, state)
        pygame.display.flip()
        
    pygame.quit()


def main() -> None:
    """Entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--ui":
        run_pygame_ui()
    else:
        future, kind, sequence = parse_arguments(sys.argv[1:])
        run_sequence_demo(sequence, future, kind)


if __name__ == "__main__":
    main()
