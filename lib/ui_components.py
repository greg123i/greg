
import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
BLUE = (0, 0, 255)
GREEN = (0, 200, 0)
RED = (255, 0, 0)

class TooltipManager:
    def __init__(self):
        self.hovered_element = None
        self.hover_start_time = 0
        self.delay = 500 # ms
        
    def update(self, element):
        if element != self.hovered_element:
            self.hovered_element = element
            self.hover_start_time = pygame.time.get_ticks()
            
    def draw(self, screen, font):
        if self.hovered_element and hasattr(self.hovered_element, 'tooltip') and self.hovered_element.tooltip:
            if pygame.time.get_ticks() - self.hover_start_time > self.delay:
                mouse_pos = pygame.mouse.get_pos()
                text = self.hovered_element.tooltip
                
                # Render text
                text_surf = font.render(text, True, (0, 0, 0))
                bg_rect = text_surf.get_rect()
                bg_rect.topleft = (mouse_pos[0] + 15, mouse_pos[1] + 15)
                
                # Expand bg
                bg_rect.width += 10
                bg_rect.height += 6
                
                # Ensure within screen
                if bg_rect.right > screen.get_width():
                    bg_rect.right = mouse_pos[0] - 5
                if bg_rect.bottom > screen.get_height():
                    bg_rect.bottom = mouse_pos[1] - 5
                    
                pygame.draw.rect(screen, (255, 255, 220), bg_rect)
                pygame.draw.rect(screen, (0, 0, 0), bg_rect, 1)
                
                screen.blit(text_surf, (bg_rect.x + 5, bg_rect.y + 3))

class Button:
    def __init__(self, x, y, width, height, text, action=None, color=DARK_GRAY, tooltip=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.color = color
        self.tooltip = tooltip
        self.is_hovered = False
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered and self.action:
                self.action()
                
    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered
        
    def draw(self, screen, font):
        color = self.color
        if self.is_hovered:
            # Lighten color
            color = tuple(min(255, c + 30) for c in self.color)
            
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=5)
        
        text_surf = font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

class InputBox:
    def __init__(self, x, y, width, height, font=None, text='', text_color=WHITE, cursor_color=WHITE, tooltip=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = GRAY
        self.text = text
        self.font = font
        self.txt_surface = None
        if self.font:
            self.txt_surface = self.font.render(text, True, text_color)
        self.active = False
        self.text_color = text_color
        self.cursor_color = cursor_color
        self.tooltip = tooltip
        self.is_hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = WHITE if self.active else GRAY
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    # Optional: Trigger something?
                    pass
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                if self.font:
                    self.txt_surface = self.font.render(self.text, True, self.text_color)

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered

    def set_text(self, text):
        self.text = text
        if self.font:
            self.txt_surface = self.font.render(self.text, True, self.text_color)

    def draw(self, screen, font=None):
        # Draw Background
        pygame.draw.rect(screen, (50, 50, 50), self.rect)
        # Blit the text.
        if self.txt_surface:
            screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 10))
        # Draw the rect.
        pygame.draw.rect(screen, self.color, self.rect, 2)
        
        # Draw Cursor
        if self.active:
            # Calculate cursor position
            if self.font:
                text_width = self.txt_surface.get_width()
                cursor_x = self.rect.x + 5 + text_width
                cursor_y = self.rect.y + 5
                pygame.draw.line(screen, self.cursor_color, (cursor_x, cursor_y), (cursor_x, cursor_y + self.rect.height - 10), 2)

class Switch:
    def __init__(self, x, y, width, height, text, initial_state=False, tooltip=None, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.state = initial_state
        self.tooltip = tooltip
        self.action = action
        self.is_hovered = False
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                self.state = not self.state
                if self.action:
                    self.action(self.state)

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered

    def draw(self, screen, font):
        # Draw Label
        label_surf = font.render(self.text, True, WHITE)
        screen.blit(label_surf, (self.rect.x, self.rect.y - 20))
        
        # Draw Switch Body
        color = GREEN if self.state else GRAY
        pygame.draw.rect(screen, color, self.rect, border_radius=self.rect.height//2)
        pygame.draw.rect(screen, WHITE, self.rect, 2, border_radius=self.rect.height//2)
        
        # Draw Knob
        knob_size = self.rect.height - 4
        knob_x = self.rect.right - knob_size - 2 if self.state else self.rect.x + 2
        knob_rect = pygame.Rect(knob_x, self.rect.y + 2, knob_size, knob_size)
        pygame.draw.ellipse(screen, WHITE, knob_rect)

class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, tooltip=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.tooltip = tooltip
        self.is_hovered = False
        self.dragging = False
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                self.dragging = True
                self.update_value(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.update_value(event.pos[0])

    def update_value(self, mouse_x):
        ratio = (mouse_x - self.rect.x) / self.rect.width
        ratio = max(0, min(1, ratio))
        self.value = self.min_val + (ratio * (self.max_val - self.min_val))

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered

    def draw(self, screen, font):
        # Draw Line
        pygame.draw.rect(screen, GRAY, self.rect)
        
        # Draw Handle
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + (ratio * self.rect.width)
        handle_rect = pygame.Rect(handle_x - 5, self.rect.y - 5, 10, self.rect.height + 10)
        pygame.draw.rect(screen, BLUE if self.is_hovered or self.dragging else WHITE, handle_rect)
        
        # Draw Value Text
        val_surf = font.render(f"{self.value:.2f}", True, WHITE)
        screen.blit(val_surf, (self.rect.right + 10, self.rect.y))
