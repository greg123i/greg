import os
import sys
import datetime

# Ensure project root is on sys.path so we can import lib.*
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    import pygame
except ImportError:
    pygame = None

from lib.ui_components import Button, InputBox, TooltipManager, WHITE, BLACK, GRAY, DARK_GRAY, BLUE, GREEN
from lib.model_interface import ModelInterface

# --- Log Capture System ---
class LogCapture:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = []
    
    def write(self, message):
        self.stream.write(message)
        self.buffer.append(message)
        if len(self.buffer) > 5000: # Keep last ~5000 chunks
            self.buffer = self.buffer[-4000:]
            
    def flush(self):
        self.stream.flush()
        
    def get_contents(self):
        return "".join(self.buffer)

# Redirect stdout/stderr
log_capture = LogCapture(sys.stdout)
sys.stdout = log_capture
sys.stderr = log_capture # Capture both into same stream for interleaved context

# Page Constants
PAGE_CHAT = "Chat"
PAGE_TRAINING = "Training"

def draw_chat_area(screen, font, chat_lines, area_rect):
    pygame.draw.rect(screen, (30, 30, 30), area_rect)
    pygame.draw.rect(screen, GRAY, area_rect, 2)
    
    line_height = font.get_linesize()
    max_width = area_rect.width - 10
    
    # Wrap text logic
    wrapped_lines = []
    for line in chat_lines:
        parts = line.split('\n')
        for part in parts:
            words = part.split(' ')
            current_line = []
            while words:
                word = words.pop(0)
                test_line = ' '.join(current_line + [word])
                if font.size(test_line)[0] <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        wrapped_lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        wrapped_lines.append(word)
                        current_line = []
            if current_line:
                wrapped_lines.append(' '.join(current_line))
    
    # Only show what fits
    max_visible_lines = (area_rect.height - 10) // line_height
    visible_lines = wrapped_lines[-max_visible_lines:]
    
    y = area_rect.y + 5
    for line in visible_lines:
        text_surf = font.render(line, True, WHITE)
        screen.blit(text_surf, (area_rect.x + 5, y))
        y += line_height


def draw_status_line(screen, font, status_text, area_rect):
    pygame.draw.rect(screen, (20, 20, 20), area_rect)
    pygame.draw.rect(screen, GRAY, area_rect, 1)
    text_surf = font.render(status_text, True, WHITE)
    screen.blit(text_surf, (area_rect.x + 5, area_rect.y + 5))


def run_gui():
    if pygame is None:
        print("Please install pygame to use the GUI.")
        return

    pygame.init()
    width = 1000
    height = 720
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("Greg AI Model GUI (Initializing...)")

    font = pygame.font.SysFont("consolas", 18)
    small_font = pygame.font.SysFont("consolas", 14)
    header_font = pygame.font.SysFont("consolas", 20, bold=True)
    
    tooltip_manager = TooltipManager()

    # Application State
    current_page = PAGE_CHAT
    
    # Navigation Buttons
    nav_btn_chat = Button(0, 0, 100, 40, "Chat", color=BLUE, tooltip="Switch to Chat Interface")
    nav_btn_train = Button(0, 0, 100, 40, "Training", color=DARK_GRAY, tooltip="Switch to Training Interface")
    nav_buttons = [nav_btn_chat, nav_btn_train]

    # Layout Rects (Initialized later)
    chat_rect = pygame.Rect(0, 0, 0, 0)
    status_rect = pygame.Rect(0, 0, 0, 0)
    training_area_rect = pygame.Rect(0, 0, 0, 0)
    
    # Chat Page Components
    input_box = InputBox(0, 0, 0, 0, font=font, tooltip="Type your message to Greg here")
    send_button = Button(0, 0, 120, 40, "Send", color=BLUE, tooltip="Send message to Greg")
    
    # Training Page Components
    # x, y, w, h
    ib_epochs = InputBox(0, 0, 80, 40, font=font, text="2", tooltip="Number of epochs to train")
    ib_mem    = InputBox(0, 0, 80, 40, font=font, text="1", tooltip="Enable Persistent Memory (1=On, 0=Off)")
    ib_batch  = InputBox(0, 0, 80, 40, font=font, text="16", tooltip="Batch size for training")
    ib_seq    = InputBox(0, 0, 80, 40, font=font, text="50", tooltip="Sequence length (context window)")
    ib_elite  = InputBox(0, 0, 80, 40, font=font, text="25", tooltip="Number of elite organisms to keep")
    ib_loss   = InputBox(0, 0, 80, 40, font=font, text="1e-5", tooltip="Target loss to stop training")
    ib_time   = InputBox(0, 0, 80, 40, font=font, text="0", tooltip="Max training time in minutes (0=Infinite)")
    
    param_boxes = [ib_epochs, ib_mem, ib_batch, ib_seq, ib_elite, ib_loss, ib_time]
    param_labels = ["Epochs", "Memory Mode", "Batch Size", "Seq Length", "Elitism", "Target Loss", "Time (min)"]
    
    train_button = Button(0, 0, 160, 40, "Train Model", color=GREEN, tooltip="Start training with current parameters")
    stop_button = Button(0, 0, 100, 40, "Stop", color=(200, 50, 50), tooltip="Stop current training session")
    fetch_button = Button(0, 0, 160, 40, "Fetch Data", color=DARK_GRAY, tooltip="Crawl web for more training data (auto-sorts)")
    sort_button = Button(0, 0, 160, 40, "Sort Data", color=DARK_GRAY, tooltip="Classify and sort data using Ollama")
    graph_button = Button(0, 0, 140, 40, "Graph: OFF", color=DARK_GRAY, tooltip="Toggle loss history graph")
    dump_button = Button(0, 0, 140, 40, "Dump Log", color=(180, 180, 50), tooltip="Save error logs to file and clipboard")
    
    training_buttons = [train_button, stop_button, fetch_button, sort_button, graph_button, dump_button]

    # Layout Update Function
    def update_layout(w, h):
        nonlocal chat_rect, status_rect, training_area_rect
        
        # Minimum size enforcement
        if w < 800: w = 800
        if h < 600: h = 600
        
        if screen.get_width() != w or screen.get_height() != h:
             pygame.display.set_mode((w, h), pygame.RESIZABLE)

        padding = 15
        header_height = 60
        status_height = 30
        
        # Header (Navigation)
        nav_x = padding
        for btn in nav_buttons:
            btn.rect.topleft = (nav_x, padding)
            nav_x += btn.rect.width + padding

        # Common Status Bar at Bottom
        status_rect.update(padding, h - status_height - padding, w - (padding * 2), status_height)
        
        content_y = header_height + padding
        content_h = h - header_height - status_height - (padding * 3)
        
        # --- Chat Page Layout ---
        input_height = 50
        send_btn_width = 100
        
        # Chat Area
        chat_h = content_h - input_height - padding
        chat_rect.update(padding, content_y, w - (padding * 2), chat_h)
        
        # Input Area
        input_y = chat_rect.bottom + padding
        input_box.rect.update(padding, input_y, w - (padding * 3) - send_btn_width, input_height)
        send_button.rect.update(input_box.rect.right + padding, input_y, send_btn_width, input_height)
        
        # --- Training Page Layout ---
        training_area_rect.update(padding, content_y, w - (padding * 2), content_h)
        
        # Responsive split: Diagnostics (40%) | Params (60%)
        diag_width = int(w * 0.4)
        params_x = padding + diag_width + 20
        
        # Param Boxes Layout
        box_width = 80
        start_y = content_y + 40
        for i, box in enumerate(param_boxes):
            box.rect.update(params_x + 120, start_y + (i * 50), box_width, 35)
            
        # Training Buttons Layout (Below params)
        btn_y = start_y + (len(param_boxes) * 50) + 20
        
        train_button.rect.topleft = (params_x, btn_y)
        stop_button.rect.topleft = (train_button.rect.right + 20, btn_y)
        
        # Row 2 (Utilities)
        btn_y_row2 = btn_y + 50
        fetch_button.rect.topleft = (params_x, btn_y_row2)
        sort_button.rect.topleft = (fetch_button.rect.right + 20, btn_y_row2)
        graph_button.rect.topleft = (sort_button.rect.right + 20, btn_y_row2)
        dump_button.rect.topleft = (graph_button.rect.right + 20, btn_y_row2)
        
        # Ensure Status Bar is at bottom
        status_rect.topleft = (0, h - 30)


    update_layout(width, height)

    model = ModelInterface()
    pygame.display.set_caption(f"Greg AI Model GUI ({model.device})")
    chat_history = []
    chat_lines = []
    status_text = f"Ready. Device: {model.device}"
    show_diag = True # Default to showing diag on training page

    def add_chat_line(text):
        nonlocal chat_lines
        chat_lines.append(text)

    # --- Actions ---
    def set_page(page):
        nonlocal current_page
        current_page = page
        # Update nav button colors
        if page == PAGE_CHAT:
            nav_btn_chat.color = BLUE
            nav_btn_train.color = DARK_GRAY
        else:
            nav_btn_chat.color = DARK_GRAY
            nav_btn_train.color = BLUE

    nav_btn_chat.action = lambda: set_page(PAGE_CHAT)
    nav_btn_train.action = lambda: set_page(PAGE_TRAINING)

    def on_send():
        nonlocal status_text
        text = input_box.text.strip()
        if not text:
            return
        add_chat_line("User: " + text)
        input_box.set_text("")
        status_text = "Greg is thinking..."
        
        # Force a redraw so user sees "Thinking..." immediately (hacky but works in single thread)
        # Actually, we can't easily force redraw without breaking the loop. 
        # But since generate is blocking, the UI will freeze. 
        # We really should run generation in a thread, but for now let's just update status.
        
        try:
            pygame.display.set_caption("Greg AI (Thinking...)")
            pygame.event.pump() # Prevent "Not Responding"
            
            response = model.generate_chat_response(chat_history, text)
            if not response:
                response = "[...]" # Visual indicator for empty response
            
            add_chat_line("AI: " + response)
            status_text = model.status_message
        except Exception as error:
            status_text = "Error: " + str(error)
        finally:
             pygame.display.set_caption(f"Greg AI Model GUI ({model.device})")

    def on_train():
        nonlocal status_text
        if model.is_training:
            status_text = "Already training."
            return
        
        try:
            epochs = int(ib_epochs.text)
            use_memory = int(ib_mem.text) > 0
            batch_size = int(ib_batch.text)
            seq_len = int(ib_seq.text)
            elitism = int(ib_elite.text)
            target_loss = float(ib_loss.text)
            time_minutes = float(ib_time.text)
        except ValueError:
            status_text = "Invalid training parameters."
            return

        model.start_training(
            epochs=epochs, 
            persistent_memory=use_memory,
            batch_size=batch_size, 
            seq_len=seq_len,
            elitism_epochs=elitism,
            target_loss=target_loss,
            time_minutes=time_minutes if time_minutes > 0 else None
        )
        status_text = f"Started training (Target: {target_loss})..."
    
    def on_stop():
        model.stop_training()

    def on_graph():
        model.show_loss_graph = not model.show_loss_graph
        graph_button.text = "Graph: ON" if model.show_loss_graph else "Graph: OFF"
        graph_button.color = GREEN if model.show_loss_graph else DARK_GRAY

    def on_dump_log():
        nonlocal status_text
        try:
            log_content = log_capture.get_contents()
            
            # 1. Write to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"greg_error_log_{timestamp}.txt"
            log_path = os.path.join(PROJECT_ROOT, fname)
            
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(log_content)
                
            # 2. Leave a message (Save to fixed 'latest_message.txt')
            # This allows the user to easily find the last log without searching through timestamps.
            message_path = os.path.join(PROJECT_ROOT, "latest_message.txt")
            with open(message_path, "w", encoding="utf-8") as f:
                f.write(log_content)

            status_text = f"Log saved to {fname} & latest_message.txt"
        except Exception as e:
            status_text = f"Dump failed: {e}"

    def on_fetch_data():
        nonlocal status_text
        if model.is_training:
            status_text = "Busy training, cannot fetch data now."
            return
        model.fetch_training_data(target_count=10000)
        status_text = "Crawling for 10k files..."

    def on_sort_data():
        nonlocal status_text
        if model.is_training:
            status_text = "Busy training, cannot sort data now."
            return
        from lib.data_tools import classify_and_sort
        status_text = "Sorting data with Ollama... Check Terminal."
        count = classify_and_sort(model.data_directory)
        status_text = f"Sorted {count} files."
    
    # Assign actions
    send_button.action = on_send
    train_button.action = on_train
    stop_button.action = on_stop
    fetch_button.action = on_fetch_data
    sort_button.action = on_sort_data
    graph_button.action = on_graph
    dump_button.action = on_dump_log

    clock = pygame.time.Clock()
    running = True

    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                update_layout(width, height)
            
            # Nav Buttons always active
            for btn in nav_buttons:
                btn.handle_event(event)
            
            # Page Specific Events
            if current_page == PAGE_CHAT:
                input_box.handle_event(event)
                send_button.handle_event(event)
            elif current_page == PAGE_TRAINING:
                for box in param_boxes:
                    box.handle_event(event)
                for btn in training_buttons:
                    btn.handle_event(event)

        # Hover Checks
        mouse_pos = pygame.mouse.get_pos()
        hovered_element = None

        for btn in nav_buttons:
            if btn.check_hover(mouse_pos):
                hovered_element = btn
            
        if current_page == PAGE_CHAT:
            if send_button.check_hover(mouse_pos):
                hovered_element = send_button
            if input_box.check_hover(mouse_pos):
                hovered_element = input_box
        elif current_page == PAGE_TRAINING:
            for btn in training_buttons:
                if btn.check_hover(mouse_pos):
                    hovered_element = btn
            for box in param_boxes:
                if box.check_hover(mouse_pos):
                    hovered_element = box

        tooltip_manager.update(hovered_element)

        # Drawing
        screen.fill((15, 15, 15))
        
        # Header Background
        pygame.draw.rect(screen, (25, 25, 25), (0, 0, width, 60))
        pygame.draw.line(screen, GRAY, (0, 60), (width, 60), 1)
        
        # Draw Nav Buttons
        for btn in nav_buttons:
            btn.draw(screen, font)

        # Draw Page Content
        if current_page == PAGE_CHAT:
            draw_chat_area(screen, font, chat_lines, chat_rect)
            input_box.draw(screen, font)
            send_button.draw(screen, font)
            
        elif current_page == PAGE_TRAINING:
            # Draw Training Params
            for idx, box in enumerate(param_boxes):
                box.draw(screen, font)
                # Draw Label (Right aligned to left of box)
                lbl = font.render(param_labels[idx] + ":", True, GRAY)
                lbl_rect = lbl.get_rect(midright=(box.rect.left - 10, box.rect.centery))
                screen.blit(lbl, lbl_rect)
            
            # Draw Buttons
            for btn in training_buttons:
                btn.draw(screen, font)
                
            # Draw Diagnostics/Logs (Left side of training page)
            diag_w = int(width * 0.4)
            diag_area_rect = pygame.Rect(
                training_area_rect.x, 
                training_area_rect.y, 
                diag_w, 
                training_area_rect.height
            )
            pygame.draw.rect(screen, (10, 10, 10), diag_area_rect)
            pygame.draw.rect(screen, DARK_GRAY, diag_area_rect, 1)
            
            # Header for diagnostics
            header_s = font.render("Training Log / Diagnostics", True, WHITE)
            screen.blit(header_s, (diag_area_rect.x + 10, diag_area_rect.y + 10))
            
            diag_lines = model.get_diagnostics()
            y = diag_area_rect.y + 40
            for line in diag_lines:
                if y + small_font.get_linesize() > diag_area_rect.bottom - 150: # Leave space for graph/prediction
                    break
                ts = small_font.render(line, True, WHITE)
                screen.blit(ts, (diag_area_rect.x + 10, y))
                y += small_font.get_linesize()

            # Draw Current Prediction (Below logs)
            if model.prediction_result:
                pred_y = diag_area_rect.bottom - 140
                p_hdr = small_font.render("Current Greg Prediction:", True, GREEN)
                screen.blit(p_hdr, (diag_area_rect.x + 10, pred_y))
                # Wrap prediction text
                p_text = model.prediction_result
                p_lines = []
                while p_text:
                    # Find how much fits in diag_w
                    for i in range(len(p_text), 0, -1):
                        if small_font.size(p_text[:i])[0] < diag_w - 20:
                            p_lines.append(p_text[:i])
                            p_text = p_text[i:]
                            break
                    if len(p_lines) > 3: break # Only show 3 lines
                
                for i, pline in enumerate(p_lines):
                    ps = small_font.render(pline, True, WHITE)
                    screen.blit(ps, (diag_area_rect.x + 10, pred_y + 20 + (i * 18)))

            # Draw Loss Graph if enabled
            if model.show_loss_graph and len(model.loss_history) > 2:
                graph_rect = pygame.Rect(diag_area_rect.x + 10, diag_area_rect.bottom - 60, diag_w - 20, 50)
                pygame.draw.rect(screen, BLACK, graph_rect)
                pygame.draw.rect(screen, GRAY, graph_rect, 1)
                
                points = []
                max_l = max(model.loss_history)
                min_l = min(model.loss_history)
                l_range = max(1e-6, max_l - min_l)
                
                history_subset = model.loss_history[-100:] # Only show last 100 points
                for i, val in enumerate(history_subset):
                    px = graph_rect.x + (i / len(history_subset)) * graph_rect.width
                    py = graph_rect.bottom - ((val - min_l) / l_range) * graph_rect.height
                    points.append((px, py))
                
                if len(points) > 1:
                    pygame.draw.lines(screen, GREEN, False, points, 2)

            # Update Window Title based on state
        if model.is_training:
            pygame.display.set_caption(f"Greg AI - TRAINING (Epoch {model.diag.get('epoch', '?')})")
        else:
            pygame.display.set_caption("Greg AI - Idle")

        # Draw Status Bar (Always on top of everything at bottom)
        draw_status_line(screen, small_font, model.status_message or status_text, status_rect)

        # Draw Tooltips (Last, on top of everything)
        tooltip_manager.draw(screen, small_font)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    run_gui()
