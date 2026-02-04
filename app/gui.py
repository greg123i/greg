import os
import sys
import datetime
import threading

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
    diag_area_rect = pygame.Rect(0, 0, 0, 0)
    controls_area_rect = pygame.Rect(0, 0, 0, 0)
    
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
    ib_lr     = InputBox(0, 0, 80, 40, font=font, text="1e-4", tooltip="Learning Rate")
    ib_smooth = InputBox(0, 0, 80, 40, font=font, text="0.9", tooltip="Smoothing Factor")
    
    param_boxes = [ib_epochs, ib_mem, ib_batch, ib_seq, ib_elite, ib_loss, ib_time, ib_lr, ib_smooth]
    param_labels = ["Epochs", "Memory Mode", "Batch Size", "Seq Length", "Elitism", "Target Loss", "Time (min)", "Learning Rate", "Smooth Factor"]
    
    train_button = Button(0, 0, 160, 40, "Train Model", color=GREEN, tooltip="Start training with current parameters")
    stop_button = Button(0, 0, 100, 40, "Stop", color=(200, 50, 50), tooltip="Stop current training session")
    fetch_button = Button(0, 0, 160, 40, "Fetch Data", color=DARK_GRAY, tooltip="Crawl web for more training data")
    graph_button = Button(0, 0, 140, 40, "Graph: OFF", color=DARK_GRAY, tooltip="Toggle loss history graph")
    dump_button = Button(0, 0, 140, 40, "Dump Log", color=(180, 180, 50), tooltip="Save error logs to file and clipboard")
    
    training_buttons = [train_button, stop_button, fetch_button, graph_button, dump_button]

    # Layout Update Function
    def update_layout(w, h):
        nonlocal chat_rect, status_rect, training_area_rect, diag_area_rect, controls_area_rect
        
        # Minimum size enforcement (Virtual Layout Size)
        # We calculate layout based on at least 800x600, so if the window is smaller,
        # the content is simply clipped (off-screen) rather than squashed/broken.
        layout_w = max(w, 800)
        layout_h = max(h, 600)
        
        # Note: Do NOT call pygame.display.set_mode here to avoid fighting the OS resize event.

        padding = 15
        header_height = 60
        status_height = 30
        
        # Header (Navigation)
        nav_x = padding
        for btn in nav_buttons:
            btn.rect.topleft = (nav_x, padding)
            nav_x += btn.rect.width + padding

        # Common Status Bar at Bottom
        status_rect.update(padding, layout_h - status_height - padding, layout_w - (padding * 2), status_height)
        
        content_y = header_height + padding
        content_h = layout_h - header_height - status_height - (padding * 3)
        
        # --- Chat Page Layout ---
        input_height = 50
        send_btn_width = 100
        
        # Chat Area
        chat_h = content_h - input_height - padding
        chat_rect.update(padding, content_y, layout_w - (padding * 2), chat_h)
        
        # Input Area
        input_y = chat_rect.bottom + padding
        input_box.rect.update(padding, input_y, layout_w - (padding * 3) - send_btn_width, input_height)
        send_button.rect.update(input_box.rect.right + padding, input_y, send_btn_width, input_height)
        
        # --- Training Page Layout ---
        training_area_rect.update(padding, content_y, layout_w - (padding * 2), content_h)
        
        # Split: Diagnostics (40%) | Controls (Remaining)
        diag_width = int(training_area_rect.width * 0.4)
        gap = 20
        controls_width = training_area_rect.width - diag_width - gap
        
        diag_area_rect.update(training_area_rect.x, training_area_rect.y, diag_width, training_area_rect.height)
        controls_area_rect.update(diag_area_rect.right + gap, training_area_rect.y, controls_width, training_area_rect.height)
        
        # Controls Layout (Inside controls_area_rect)
        
        # Params
        box_width = 80
        label_width = 120 # Approx
        row_height = 45
        
        # Center the block of params
        content_block_w = label_width + box_width
        start_x = controls_area_rect.x + (controls_area_rect.width - content_block_w) // 2 + label_width
        start_y = controls_area_rect.y + 40
        
        for i, box in enumerate(param_boxes):
            box.rect.update(start_x, start_y + (i * row_height), box_width, 35)
            
        # Buttons (Below params) - Flow Layout
        btn_y = start_y + (len(param_boxes) * row_height) + 30
        
        current_x = controls_area_rect.x + 20
        current_y = btn_y
        
        for btn in training_buttons:
            # Check if button fits in current row
            if current_x + btn.rect.width > controls_area_rect.right - 20:
                # Move to next row
                current_x = controls_area_rect.x + 20
                current_y += 50
            
            btn.rect.topleft = (current_x, current_y)
            current_x += btn.rect.width + 20
        
        # Ensure Status Bar is at bottom (update again if needed, but it's fixed relative to layout_h)
        status_rect.topleft = (padding, layout_h - status_height - padding)


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
            
            # Parse new parameters (Assuming user might add inputs later, but hardcoding defaults for now or reading from hidden/new fields)
            # Since user asked to "make it parameters", I should ideally add InputBoxes.
            # For now, I'll read from new InputBoxes I'll create below, or just defaults if not present.
            # But wait, I haven't created the InputBoxes in the GUI layout yet.
            # I should add them.
            
            # Let's assume standard defaults if UI elements aren't there, 
            # but I will add them to the UI layout in a moment.
            initial_lr = float(ib_lr.text) if 'ib_lr' in globals() else 1e-4
            smoothing = float(ib_smooth.text) if 'ib_smooth' in globals() else 0.9
            
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
            time_minutes=time_minutes if time_minutes > 0 else None,
            initial_lr=initial_lr,
            smoothing_factor=smoothing
        )
        status_text = f"Started training (Target: {target_loss})..."
    
    def on_stop():
        model.stop_training()

    def on_graph():
        model.graph_mode = (model.graph_mode + 1) % 3
        modes = ["OFF", "SMOOTH", "RAW"]
        colors = [DARK_GRAY, GREEN, (200, 100, 50)]
        model.show_loss_graph = (model.graph_mode > 0)
        graph_button.text = f"Graph: {modes[model.graph_mode]}"
        graph_button.color = colors[model.graph_mode]

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

            # 3. Export Diagnostics
            diag_file = model.export_diagnostics()
            
            status_text = f"Log saved. Diagnostics: {os.path.basename(diag_file)}"
        except Exception as e:
            status_text = f"Dump failed: {e}"

    def on_fetch_data():
        nonlocal status_text
        if model.is_training:
            status_text = "Busy training, cannot fetch data now."
            return
        model.fetch_training_data(target_count=10000)
        status_text = "Crawling for 10k files..."

    # Assign actions
    send_button.action = on_send
    train_button.action = on_train
    stop_button.action = on_stop
    fetch_button.action = on_fetch_data
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
            # --- Draw Background Islands ---
            
            # Diagnostics Island (Left)
            pygame.draw.rect(screen, (10, 10, 10), diag_area_rect)
            pygame.draw.rect(screen, DARK_GRAY, diag_area_rect, 1)
            
            # Controls Island (Right)
            pygame.draw.rect(screen, (10, 10, 10), controls_area_rect)
            pygame.draw.rect(screen, DARK_GRAY, controls_area_rect, 1)
            
            # --- Draw Controls Content (Right Island) ---
            
            # Params
            for idx, box in enumerate(param_boxes):
                box.draw(screen, font)
                
                # Draw Label
                lbl = font.render(param_labels[idx] + ":", True, GRAY)
                lbl_rect = lbl.get_rect(midright=(box.rect.left - 10, box.rect.centery))
                screen.blit(lbl, lbl_rect)
                
            # Buttons
            for btn in training_buttons:
                btn.draw(screen, font)

            # --- Draw Diagnostics Content (Left Island) ---
            
            # Header for diagnostics
            header_s = font.render("Training Log / Diagnostics", True, WHITE)
            screen.blit(header_s, (diag_area_rect.x + 10, diag_area_rect.y + 10))
            
            diag_lines = model.get_diagnostics()
            y = diag_area_rect.y + 40
            
            # Determine available space for logs (leave space for prediction + graph)
            logs_bottom_limit = diag_area_rect.bottom - 150 
            
            for line in diag_lines:
                if y + small_font.get_linesize() > logs_bottom_limit:
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
                        if small_font.size(p_text[:i])[0] < diag_area_rect.width - 20:
                            p_lines.append(p_text[:i])
                            p_text = p_text[i:]
                            break
                    if len(p_lines) > 3: break # Only show 3 lines
                
                for i, pline in enumerate(p_lines):
                    ps = small_font.render(pline, True, WHITE)
                    screen.blit(ps, (diag_area_rect.x + 10, pred_y + 20 + (i * 18)))

            # Draw Loss Graph if enabled
            data_source = model.loss_history if model.graph_mode == 1 else model.raw_loss_history
            if model.show_loss_graph:
                graph_rect = pygame.Rect(diag_area_rect.x + 10, diag_area_rect.bottom - 60, diag_area_rect.width - 20, 50)
                pygame.draw.rect(screen, BLACK, graph_rect)
                pygame.draw.rect(screen, GRAY, graph_rect, 1)

                if len(data_source) > 2:
                    history_subset = data_source[-100:]
                    if not history_subset: history_subset = [0]
                    
                    current_max = max(history_subset)
                    current_min = min(history_subset)
                    
                    # Update Smoothed View Bounds for Stability
                    if model.graph_view_max is None:
                         model.graph_view_max = current_max
                         model.graph_view_min = current_min
                    else:
                         # Slow decay/adapt (5%) to prevent sea-sawing
                         model.graph_view_max = model.graph_view_max * 0.95 + current_max * 0.05
                         model.graph_view_min = model.graph_view_min * 0.95 + current_min * 0.05
                    
                    l_range = max(1e-6, model.graph_view_max - model.graph_view_min)
                    base_min = model.graph_view_min
                    
                    points = []
                    for i, val in enumerate(history_subset):
                        px = graph_rect.x + (i / max(1, len(history_subset)-1)) * graph_rect.width
                        
                        # Calculate Y based on smoothed bounds
                        norm_y = (val - base_min) / l_range
                        py = graph_rect.bottom - (norm_y * graph_rect.height)
                        points.append((px, py))
                    
                    # Clip drawing to graph area
                    old_clip = screen.get_clip()
                    screen.set_clip(graph_rect)
                    
                    line_color = GREEN if model.graph_mode == 1 else (200, 100, 50)
                    if len(points) > 1:
                        pygame.draw.lines(screen, line_color, False, points, 2)
                        
                    screen.set_clip(old_clip)

                    # Interactive Tooltip
                    if graph_rect.collidepoint(mouse_pos):
                        rel_x = mouse_pos[0] - graph_rect.x
                        hover_idx = int((rel_x / graph_rect.width) * len(history_subset))
                        hover_idx = max(0, min(len(history_subset)-1, hover_idx))
                        
                        if hover_idx < len(points):
                            hx, hy = points[hover_idx]
                            val = history_subset[hover_idx]
                            
                            # Clamp marker Y to rect
                            hy = max(graph_rect.top, min(graph_rect.bottom, hy))
                            
                            # Draw Marker
                            pygame.draw.circle(screen, WHITE, (int(hx), int(hy)), 4)
                            pygame.draw.line(screen, (100, 100, 100), (int(hx), graph_rect.top), (int(hx), graph_rect.bottom), 1)
                            
                            # Draw Tooltip
                            tip_text = f"L: {val:.4f}"
                            ts = small_font.render(tip_text, True, WHITE)
                            tip_rect = ts.get_rect(bottomleft=(int(hx), int(hy) - 10))
                            
                            # Clamp tooltip
                            if tip_rect.right > width - 10: tip_rect.right = width - 10
                            if tip_rect.left < 10: tip_rect.left = 10
                            
                            pygame.draw.rect(screen, (40, 40, 40), tip_rect.inflate(8, 4))
                            pygame.draw.rect(screen, WHITE, tip_rect.inflate(8, 4), 1)
                            screen.blit(ts, tip_rect)
                else:
                    # No Data Message
                    nd_text = small_font.render("Waiting for data...", True, GRAY)
                    nd_rect = nd_text.get_rect(center=graph_rect.center)
                    screen.blit(nd_text, nd_rect)

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
